rm(list = ls())

# Load libraries
library(tidyverse) #For loading CSV file
library(caTools) #For splitting into train and test
library(ggplot2)
library(reshape2)

# Set random seed
set.seed(999)

###################################################
# Load data
###################################################

features_file_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/pupilometry_features.csv"
outcome_file_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/ICU_outcome.csv"
features <- read_csv(features_file_path, show_col_types = FALSE) %>% select(-`...1`) %>% select(-`date_examination_merged`)
ICU_outcome <-  read_csv(outcome_file_path, show_col_types = FALSE) %>% select(record_id, ICU_outcome) %>% mutate(ICU_outcome = ifelse(ICU_outcome == "A", 1, 0))#Rename ICU outcome to 0/1


# Combine data to allow for splitting it
combined_data <- inner_join(features, ICU_outcome, by = "record_id")

# Remove rows with ANY missing values
clean_data <- combined_data %>% drop_na()

#Extracting first day data
first_day_data <- clean_data[clean_data$redcap_repeat_instance == 1, ]
head(first_day_data)

split <- sample.split(first_day_data$ICU_outcome, SplitRatio = 0.8)

training_data <- subset(first_day_data, split == "TRUE")
testing_data <- subset(first_day_data, split == "FALSE")

###################################################
# Center and scale data using training data values
###################################################

cols_to_scale <- c("left_arousal_gradient", "left_max_PLR", "left_LOR_early_gradient", "left_LOR_late_gradient", "left_50pct_PLR_time", "left_50pct_LOR_time")

# Compute scaling parameters
scaling_params <- training_data %>% summarize(across(all_of(cols_to_scale), list(mean = ~mean(.), sd = ~sd(.))))

#Scaling training data
training_data_scaled <- training_data %>% mutate(across(all_of(cols_to_scale), ~scale(.) %>% as.vector()))

#Scaling testing data using training parameters
testing_data_scaled <- testing_data %>% mutate(across(all_of(cols_to_scale), ~(. - mean(training_data[[cur_column()]],)) / sd(training_data[[cur_column()]])))


###################################################
# Modelling
###################################################
#Construct model
logistic_model <- glm(ICU_outcome ~ left_arousal_gradient + left_max_PLR + left_LOR_early_gradient + left_LOR_late_gradient + left_50pct_PLR_time + left_50pct_LOR_time,
                      data = training_data_scaled,
                      family = binomial(link = "logit"))
logistic_model

summary(logistic_model)

#Predict test data
predict_prob <- predict(logistic_model,
                       testing_data_scaled, type = "response")

# Convert predictions to class predictions using 0.5 threshold
predict_class <- ifelse(predict_prob > 0.5, 1, 0)

predictions <- data.frame(
                          probability = predict_prob,
                          predicted_class = predict_class,
                          actual_class = testing_data_scaled$ICU_outcome
                          )


###################################################
# Evaluation
###################################################
# Creating confusion matrix
conf_matrix <- table(Predicted = predict_class, Actual = testing_data_scaled$ICU_outcome)

# Extracting evaluation metrics
TP <- conf_matrix[2, 2]
TN <- conf_matrix[1, 1]
FP <- conf_matrix[2, 1]
FN <- conf_matrix[1, 2]

# Compute evaluation metrics
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
accuracy <- (TP + TN) / (TP + TN + FP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("Accuracy:", accuracy, "\n")
cat("F1 Score:", f1_score, "\n")

# Reshape the confusion matrix for ggplot2
conf_matrix_melted <- as.data.frame(conf_matrix)
colnames(conf_matrix_melted) <- c("Actual", "Predicted", "Count")

ggplot(conf_matrix_melted, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "black", size = 6) +  # Add text labels
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted") +
  theme_minimal()