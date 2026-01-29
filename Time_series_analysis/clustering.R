rm(list = ls())

# Load libraries
library(dplyr)
library(tidyverse)
library(dtwclust)
library(proxy)
library(cluster)
library(factoextra)
library(patchwork)
library(purrr)
library(networkD3)
library(tidyr)
library(ggalluvial)
library(ggplot2)


# Define save path
save_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Plots/Clustering"
load_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/Day_data"

#### Loading and preparing the data ####
ICU_outcome <- read_csv("L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/ICU_outcome.csv", show_col_types = FALSE)
ICU_outcome <- ICU_outcome %>% select(-...1)


run_day_clustering <- function(day,
                               input_dir  = load_path,
                               output_dir = save_path,
                               k = 3) {
  
  message("Running clustering for day ", day)
  
  # -----------------------------
  # Load data
  # -----------------------------
  file_path <- file.path(input_dir,
                         paste0("day", day, "_left_raw_data.csv"))
  
  left_eye_data <- read_csv(file_path, show_col_types = FALSE)
  colnames(left_eye_data)[1] <- "Time"
  
  
  # -----------------------------
  # DTW clustering
  # -----------------------------
  data_t <- left_eye_data %>%
    select(-Time) %>%
    t() %>%
    as.data.frame()
  
  dtw_dist <- proxy::dist(data_t, method = "dtw")
  hc <- hclust(dtw_dist, method = "ward.D2")
  clusters <- cutree(hc, k = k)
  
  clustered_data <- data.frame(
    Eye = colnames(left_eye_data)[-1],
    Cluster = clusters
  )
  
  
  
  write.csv(
    clustered_data,
    file.path(output_dir, paste0("dtw_clusters_day", day, ".csv")),
    row.names = FALSE
  )
  
  # -----------------------------
  # Long format for plotting
  # -----------------------------
  data_long <- left_eye_data %>%
    pivot_longer(cols = -Time,
                 names_to = "Eye",
                 values_to = "Pupil_Diameter") %>%
    left_join(clustered_data, by = "Eye")
  
  summary_data <- data_long %>%
    group_by(Time, Cluster) %>%
    summarise(
      Mean = mean(Pupil_Diameter, na.rm = TRUE),
      SD   = sd(Pupil_Diameter, na.rm = TRUE),
      .groups = "drop"
    )
  
  # -----------------------------
  # Figures
  # -----------------------------
  fig1 <- ggplot(data_long,
                 aes(Time, Pupil_Diameter,
                     group = Eye,
                     color = factor(Cluster))) +
    geom_line(alpha = 0.3) +
    labs(title = paste("Day", day, "– Raw Reflexes")) +
    theme_minimal() +
    theme(legend.position = "none")
  
  fig2 <- ggplot(summary_data,
                 aes(Time, Mean,
                     color = factor(Cluster),
                     fill  = factor(Cluster))) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymin = Mean - SD,
                    ymax = Mean + SD),
                alpha = 0.2) +
    labs(title = paste("Day", day, "– Mean ± SD")) +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(
    filename = file.path(output_dir,
                         paste0("day", day, "_clusters.jpg")),
    plot = fig1 + fig2,
    width = 12, height = 6, dpi = 600
  )
  
  # -----------------------------
  # Silhouette
  # -----------------------------
  sil <- silhouette(clusters, as.dist(dtw_dist))
  
  fig_sil <- fviz_silhouette(sil) +
    labs(title = paste("Day", day, "– Silhouette"))
  
  ggsave(
    filename = file.path(output_dir,
                         paste0("day", day, "_silhouette.jpg")),
    plot = fig_sil,
    width = 8, height = 6, dpi = 600
  )
  
  invisible(list(
    hc = hc,
    clusters = clustered_data,
    silhouette = sil
  ))
}
results <- lapply(1:14, run_day_clustering)


# ============================================================
# Build longitudinal cluster trajectories
# ============================================================

n_days <- length(results)

cluster_long <- map2_dfr(
  results,
  seq_len(n_days),
  ~ .x$clusters %>%
    mutate(Day = .y) %>%
    rename(record_id = Eye)
) %>%
  mutate(
    record_id = as.numeric(record_id),
    State     = paste0("C", Cluster)
  )

# ------------------------------------------------------------
# Wide format (per-patient timeline)
# ------------------------------------------------------------
cluster_wide <- cluster_long %>%
  select(record_id, Day, State) %>%
  pivot_wider(
    names_from   = Day,
    values_from  = State,
    names_prefix = "Day"
  ) %>%
  left_join(ICU_outcome, by = "record_id") %>%
  filter(!is.na(Day1))

day_cols <- paste0("Day", seq_len(n_days))

# Fill forward to handle isolated missing days
cluster_wide <- cluster_wide %>%
  arrange(record_id) %>%
  group_by(record_id) %>%
  tidyr::fill(all_of(day_cols), .direction = "down") %>%
  ungroup()

# ------------------------------------------------------------
# Absorbing outcome logic
# ------------------------------------------------------------
absorb_outcome <- function(states, outcome) {
  na_runs <- rle(is.na(states))
  ends    <- cumsum(na_runs$lengths)
  starts  <- ends - na_runs$lengths + 1
  
  idx <- which(na_runs$values & na_runs$lengths >= 2)[1]
  
  if (!is.na(idx) && !is.na(outcome)) {
    absorb_pos <- starts[idx]
    states[absorb_pos] <- ifelse(outcome == 1, "Survived", "Dead")
    states[(absorb_pos + 1):length(states)] <- NA
  }
  
  states
}

cluster_wide_absorbed <- cluster_wide %>%
  rowwise() %>%
  mutate(
    new_states = list(
      absorb_outcome(
        c_across(all_of(day_cols)),
        ICU_outcome
      )
    )
  ) %>%
  ungroup() %>%
  select(record_id, new_states) %>%
  tidyr::unnest_wider(new_states, names_sep = "")


colnames(cluster_wide_absorbed)[-1] <- day_cols

# ------------------------------------------------------------
# Back to long format (CRITICAL STEP)
# ------------------------------------------------------------
cluster_long_final <- cluster_wide_absorbed %>%
  pivot_longer(
    cols      = all_of(day_cols),
    names_to  = "Day",
    values_to = "State"
  ) %>%
  mutate(
    Day = as.integer(sub("Day", "", Day))
  )

# ------------------------------------------------------------
# Enforce absorbing states (truncate trajectories)
# ------------------------------------------------------------
terminal_day <- cluster_long_final %>%
  filter(State %in% c("Dead", "Survived")) %>%
  group_by(record_id) %>%
  summarise(terminal_day = min(Day), .groups = "drop")

cluster_long_final <- cluster_long_final %>%
  left_join(terminal_day, by = "record_id") %>%
  filter(is.na(terminal_day) | Day <= terminal_day) %>%
  select(-terminal_day) %>%
  filter(!is.na(State))   # <- ESSENTIAL for ggalluvial

alluvial_long <- cluster_long_final %>%
  mutate(
    Day = factor(Day, levels = seq_len(n_days)),
    axis = paste0("Day", Day)
  )

# ============================================================
# Sankey / Alluvial plot
# ============================================================

state_colors <- c(
  "C1" = "#4E79A7",   # muted blue
  "C2" = "#59A14F",   # muted green
  "C3" = "#F28E2B",   # muted orange
  "Dead" = "#E15759", # soft red
  "Survived" = "#76B7B2" # teal
)

ggplot(
  alluvial_long,
  aes(
    x        = Day,
    stratum  = State,
    alluvium = record_id,
    y        = 1,
    fill     = State
  )
) +
  geom_flow(alpha = 0.65, width = 0.25) +
  geom_stratum(color = "grey30", width = 0.3) +
  geom_text(
    stat = "stratum",
    aes(label = after_stat(stratum)),
    size = 2.5
  ) +
  scale_fill_manual(values = state_colors) +
  scale_x_discrete(
    limits = as.character(seq_len(n_days)),
    labels = paste("Day", seq_len(n_days)),
    expand = c(.05, .05)
  ) +
  labs(
    x    = "ICU timeline",
    y    = "Number of patients",
    fill = "State"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    legend.position    = "right"
  )


