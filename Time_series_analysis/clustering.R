rm(list = ls())

# Load libraries
library(dplyr)
library(tidyverse)
  library(dtwclust)
library(proxy)


#### Loading and preparing the data ####

# Loading day 1 data
left_eye_data <- read_csv("L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation/day1_left_raw_data.csv")

# Rename first column as 'Time' and keep it as the key for merging
colnames(left_eye_data)[1] <- "Time"

# Check structure of the datasets
head(left_eye_data)

#### Clustering analysis #####

# Remove the Time column before clustering
data_t <- left_eye_data %>% select(-Time) %>% t() %>% as.data.frame()

# Compute DTW distance matrix
dtw_dist <- proxy::dist(data_t, method = "dtw")

# Perform hierarchical clustering using Ward's method
hc <- hclust(dtw_dist, method = "ward.D2")

# Plot dendrogram
plot(hc, main = "Hierarchical Clustering with DTW", sub = "", xlab = "")

# Cut tree into 3 clusters
clusters <- cutree(hc, k = 3)

# Add cluster labels to a new dataframe
clustered_data <- data.frame(Eye = colnames(merged_data)[-1], Cluster = clusters)

# Save cluster assignments
write.csv(clustered_data, "dtw_clusters.csv", row.names = FALSE)

# Print cluster summary
table(clusters)

#### R Code for Visualization ####

# Load necessary libraries
library(ggplot2)
library(reshape2)
library(tidyr)

# Convert data to long format for ggplot
data_long <- merged_data %>%
  pivot_longer(cols = -Time, names_to = "Eye", values_to = "Pupil_Diameter")


# Merge with cluster labels
clustered_data$Eye <- as.character(clustered_data$Eye) # Ensure matching column types
data_long <- merge(data_long, clustered_data, by = "Eye")

# Plot all time series, colored by cluster
fig1 <- ggplot(data_long, aes(x = Time, y = Pupil_Diameter, group = Eye, color = as.factor(Cluster))) +
  geom_line(alpha = 0.3) +
  labs(title = "Clustered Reflexes (Raw)", x = "Time (s)", y = "Pupil Diameter (mm)", color = "Cluster") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 18),   # Larger (& bold title: face = "bold")
    axis.title.x = element_text(size = 16),               # Bigger x-axis label
    axis.title.y = element_text(size = 16),               # Bigger y-axis label
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = ("none")
  )

fig1

# Compute mean ± confidence interval per cluster
summary_data <- data_long %>%
  group_by(Time, Cluster) %>%
  summarise(Mean = mean(Pupil_Diameter, na.rm = TRUE),
            SD = sd(Pupil_Diameter, na.rm = TRUE),
            .groups = 'drop')

# Plot cluster averages with confidence bands
fig2 <- ggplot(summary_data, aes(x = Time, y = Mean, color = as.factor(Cluster))) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = Mean - SD, ymax = Mean + SD, fill = as.factor(Cluster)), alpha = 0.2) +
  labs(title = "Clustered Reflexes (Mean)", x = "Time (s)", y = "Pupil Diameter (mm)", color = "Cluster", fill = "Cluster") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 18),   # Larger (& bold title: face = "bold")
    axis.title.x = element_text(size = 16),               # Bigger x-axis label
    axis.title.y = element_text(size = 16),               # Bigger y-axis label
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = ("none")
  )

fig2 

# Load required packages
library(ggplot2)
library(ggdendro)
library(dplyr)

# Assign cluster labels (same number as k from DTW clustering)
k <- 3  # Adjust based on your chosen number of clusters
cluster_labels <- cutree(hc, k = k) 

# Convert hierarchical clustering to dendrogram format
hc_dendro <- as.dendrogram(hc)

# Extract dendrogram data
dendro_data <- ggdendro::dendro_data(hc_dendro, type = "rectangle")

# Convert cluster labels into a data frame for ggplot
labels_df <- data.frame(
  label = names(cluster_labels), 
  cluster = factor(cluster_labels)  # Convert to factor for coloring
)

# Merge cluster labels with dendrogram data
dendro_data$labels <- dendro_data$labels %>%
  left_join(labels_df, by = "label")

# Define colors (same as in your previous visualizations)
cluster_colors <- c("1" = "red", "2" = "green", "3" = "blue")  

fig3 <- ggplot() +
  geom_segment(data = dendro_data$segments, 
               aes(x = x, y = y, xend = xend, yend = yend), color = "black") +
  geom_text(data = dendro_data$labels, 
            aes(x = x, y = y, label = "o", color = cluster), 
            angle = 90, hjust = 1, size = 6) +  # Rotate & increase text size
  scale_color_manual(values = cluster_colors) +
  expand_limits(y = max(dendro_data$segments$y) * 0.1) +  # Reduce Y-axis height dynamically
  theme_minimal() +
  labs(title = "Clustering Dendrogram", 
       x = "Pupillary reflex time series", y = "Height", color = "Cluster") +
  theme(
    plot.title = element_text(size = 18),   # Larger (& bold title: face = "bold")
    axis.title.x = element_text(size = 16),               # Bigger x-axis label
    axis.title.y = element_text(size = 16),               # Bigger y-axis label
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = ("none")
  )

fig3

#### Silhouette score ####

# Load necessary libraries
library(cluster)
library(factoextra)

# Compute DTW distance matrix again (if not already computed)
#dtw_dist <- TSClusters::dtwDist(merged_data)

# Compute hierarchical clustering
hc <- hclust(as.dist(dtw_dist), method = "ward.D2")  

# Cut tree to get cluster assignments (assuming 3 clusters) #
cluster_assignments <- cutree(hc, k = 3)  # set k = x if assuming x clusters, etc.

# Compute silhouette scores
silhouette_scores <- silhouette(cluster_assignments, as.dist(dtw_dist))

# Plot silhouette scores 3 clusters
fig4 <- fviz_silhouette(silhouette_scores) + 
  labs(title = "Silhouette Plot", 
       x = (""), y = "Silhouette width (Si)", color = "Cluster") +
  theme(
    plot.title = element_text(size = 18),   # Larger (& bold title: face = "bold")
    axis.title.x = element_text(size = 16),               # Bigger x-axis label
    axis.title.y = element_text(size = 16),               # Bigger y-axis label
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = ("none")
  ) +
  annotate("text", x = 160, y = -0.3, label = "Cluster 1: average score 0.73\nCluster 2: average score 0.44\nCluster 3: average score 0.72", size = 3, hjust = 0)  # Add custom text

fig4

fig2a <- ggplot(summary_data, aes(x = Time, y = Mean, color = as.factor(Cluster))) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = Mean - SD, ymax = Mean + SD, fill = as.factor(Cluster)), alpha = 0.2) +
  labs(title = "Clustered Pupillary Reflexes (Mean ± SD)", x = "Time (s)", y = "Pupil Diameter (mm)", color = "Cluster", fill = "Cluster") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 18),   # Larger (& bold title: face = "bold")
    axis.title.x = element_text(size = 16),               # Bigger x-axis label
    axis.title.y = element_text(size = 16),               # Bigger y-axis label
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = ("bottom")
  )

library(patchwork)

samlet_fig <- (fig1+fig2) / (fig3+fig4)
samlet_fig

#build legend

manual_legend <- ggplot() +
  geom_point(aes(x = 1.3, y = 1), color = "red", size = 4) +
  geom_text(aes(x = 1.4, y = 1, label = "Cluster 1", face = "bold"), hjust = 0) +
  
  geom_point(aes(x = 1.9, y = 1), color = "green", size = 4) +
  geom_text(aes(x = 2, y = 1, label = "Cluster 2", face = "bold"), hjust = 0) +
  
  geom_point(aes(x = 2.5, y = 1), color = "blue", size = 4) +
  geom_text(aes(x = 2.6, y = 1, label = "Cluster 3", face = "bold"), hjust = 0) +
  
  coord_cartesian(xlim = c(0.8, 3.4), ylim = c(0.8, 1.2), clip = "off") +
  theme_void()


final_fig <- samlet_fig / manual_legend +
  plot_layout(heights = c(1, 1, 0.15))

final_fig

# Save
ggsave("Hierarchical Clustering.jpg", final_fig, width = 14, height = 9, dpi = 300)