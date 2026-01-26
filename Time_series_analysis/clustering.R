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


# Define save path
save_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Plots/Clustering"
load_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/Day_data"

#### Loading and preparing the data ####
ICU_outcome <- read_csv("L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/ICU_outcome.csv", show_col_types = FALSE)

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
  ) %>%
    mutate(Eye = as.numeric(Eye))
  
  stopifnot(!any(is.na(clustered_data$Eye)))
  
  
  
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
results <- lapply(1:3, run_day_clustering)

library(dplyr)
library(purrr)

cluster_long <- map2_dfr(
  results,
  1:3,
  ~ .x$clusters %>%
    mutate(Day = .y) %>%
    rename(record_id = Eye)
)

cluster_long <- cluster_long %>%
  left_join(ICU_outcome, by = "record_id")


last_day <- cluster_long %>%
  group_by(record_id) %>%
  summarise(last_day = max(Day), .groups = "drop")

cluster_long <- cluster_long %>%
  left_join(last_day, by = "record_id")

cluster_long <- cluster_long %>%
  mutate(
    State = paste0("C", Cluster),
    State = ifelse(
      Day < last_day,
      State,
      ifelse(ICU_outcome == "D", "Dead", "Survived")
    )
  )

library(tidyr)

cluster_wide <- cluster_long %>%
  select(record_id, Day, State) %>%
  pivot_wider(
    names_from  = Day,
    values_from = State,
    names_prefix = "Day"
  ) %>%
  mutate(Freq = 1)

library(ggalluvial)
library(ggplot2)

ggplot(
  cluster_wide,
  aes(
    axis1 = Day1,
    axis2 = Day2,
    axis3 = Day3,
    y = Freq
  )
) +
  geom_alluvium(aes(fill = Day1), alpha = 0.7, width = 0.2) +
  geom_stratum(width = 0.25, color = "grey30") +
  geom_text(
    stat = "stratum",
    aes(label = after_stat(stratum)),
    size = 3
  ) +
  scale_x_discrete(
    limits = c("Day 1", "Day 2", "Day 3"),
    expand = c(.05, .05)
  ) +
  labs(
    x = "ICU Day",
    y = "Number of patients",
    fill = "Initial cluster"
  ) +
  theme_minimal(base_size = 11)
