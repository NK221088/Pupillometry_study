rm(list = ls())

# Load libraries
library(dplyr)
library(tidyverse)
library(dtwclust)
library(proxy)

# Define save path
save_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Plots/Clustering"
load_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/day_data"

#### Loading and preparing the data ####

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
results <- lapply(6:20, run_day_clustering)