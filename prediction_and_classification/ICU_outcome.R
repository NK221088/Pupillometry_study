rm(list = ls())

# Load libraries
library(tidyverse) #For loading CSV file
library(caTools) #For splitting into train and test
library(ggplot2)
library(reshape2)
library(pROC)
library(lme4)

# Set random seed
set.seed(999)

###################################################
# Load data
###################################################

# features_file_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/pupilometry_features.csv"
features_file_path <- "C:/Users/NTres/OneDrive - Danmarks Tekniske Universitet/Arbejde_Rigshospitalet/Pupillometry/Data/Extracted_features/pupilometry_features.csv"
# outcome_file_path <- "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/ICU_outcome.csv"
outcome_file_path <- "C:/Users/NTres/OneDrive - Danmarks Tekniske Universitet/Arbejde_Rigshospitalet/Pupillometry/Data/Extracted_features/ICU_outcome.csv"

features <- read_csv(features_file_path, show_col_types = FALSE) %>% select(-`...1`) %>% select(-`date_examination_merged`)
ICU_outcome <-  read_csv(outcome_file_path, show_col_types = FALSE) %>% select(record_id, ICU_outcome) %>% mutate(ICU_outcome = ifelse(ICU_outcome == "A", 1, 0))#Rename ICU outcome to 0/1

# Define feature names
cols_to_scale <- c("left_arousal_gradient", "left_max_PLR", "left_LOR_early_gradient", "left_LOR_late_gradient", "left_50pct_PLR_time", "left_50pct_LOR_time")


# Combine data to allow for splitting it
combined_data <- inner_join(features, ICU_outcome, by = "record_id")

# Remove rows with ANY missing values
clean_data <- combined_data %>% drop_na()

#Extracting first day data
first_day_data <- clean_data[clean_data$redcap_repeat_instance == 1, ]
head(first_day_data)

# Computing change to day two¨
second_day_data <- clean_data[clean_data$redcap_repeat_instance == 2, ]

day_two_change <- merge(
  first_day_data,
  second_day_data,
  by = "record_id",
  suffixes = c("_day1", "_day2")
)
for (col in cols_to_scale) {
  day_two_change[[paste0(col, "_change")]] <- day_two_change[[paste0(col, "_day2")]] - day_two_change[[paste0(col, "_day1")]]
}
day_two_change <- day_two_change %>% select(record_id, ICU_outcome_day1, ends_with("_change"))

day_two_change <- day_two_change %>% rename_with(~ gsub("_day1", "", .x))

###################################################
# glmer investigation
###################################################

first_day_data_scaled <- first_day_data %>% mutate(across(all_of(cols_to_scale), scale))
first_day_data_scaled <- first_day_data_scaled %>% mutate(redcap_repeat_instance = redcap_repeat_instance - 1)

glm_first_day_model = glm(
                            ICU_outcome ~
                              left_arousal_gradient + left_max_PLR +
                              left_LOR_early_gradient + left_LOR_late_gradient +
                              left_50pct_PLR_time + left_50pct_LOR_time,
                            data = first_day_data_scaled,
                            family = binomial(link = "logit")
                            )

summary(glm_first_day_model)

change_cols <- grep("_change$", names(day_two_change), value=TRUE)
day_two_change <- day_two_change %>% mutate(across(all_of(change_cols), scale))
logistic_change_model <- glm(ICU_outcome ~
                               left_arousal_gradient_change +
                               left_max_PLR_change +
                               left_LOR_early_gradient_change +
                               left_LOR_late_gradient_change +
                               left_50pct_PLR_time_change +
                               left_50pct_LOR_time_change,
                             data = day_two_change,
                             family = binomial(link = "logit"))
logistic_change_model

summary(logistic_change_model)

###################################################
# Preparing training and test data
###################################################

split <- sample.split(first_day_data$ICU_outcome, SplitRatio = 0.8)

training_data <- subset(first_day_data, split == "TRUE")
testing_data <- subset(first_day_data, split == "FALSE")


###################################################
# Center and scale data using training data values
###################################################


# Compute scaling parameters
scaling_params <- training_data %>% summarize(across(all_of(cols_to_scale), list(mean = ~mean(.), sd = ~sd(.))))

#Scaling training data
training_data_scaled <- training_data %>% mutate(across(all_of(cols_to_scale), ~scale(.) %>% as.vector()))

#Scaling testing data using training parameters
testing_data_scaled <- testing_data %>% mutate(across(all_of(cols_to_scale), ~(. - mean(training_data[[cur_column()]],)) / sd(training_data[[cur_column()]])))

###################################################
# Modelling
###################################################
#Construct model logistic regression, using first day data
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
# Baseline model: Majority class classifier
###################################################

# Compute class probability from training data
majority_classifier <- sum(training_data_scaled$ICU_outcome) / length(training_data_scaled$ICU_outcome)
majority_class <- ifelse(majority_classifier > 0.5, 1, 0)

# Define classes
majority_predictions <- rep(majority_class, times = length(testing_data_scaled$ICU_outcome))

###################################################
# Evaluation
###################################################
#Plot the distribution of the outcome variable
barplot(table(training_data_scaled$ICU_outcome),
        xlab = "ICU outcome",
        ylab = "Count",
        main = "Distribution of ICU outcome")

# Creating confusion matrix
log_reg_conf_matrix <- table(Predicted = predict_class, Actual = testing_data_scaled$ICU_outcome)

# Reshape the confusion matrix for ggplot2
conf_matrix_melted <- as.data.frame(log_reg_conf_matrix)
colnames(conf_matrix_melted) <- c("Actual", "Predicted", "Count")

ggplot(conf_matrix_melted, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "black", size = 6) +  # Add text labels
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted") +
  theme_minimal()

# Extracting evaluation metrics
log_reg_TP <- log_reg_conf_matrix[2, 2]
log_reg_TN <- log_reg_conf_matrix[1, 1]
log_reg_FP <- log_reg_conf_matrix[2, 1]
log_reg_FN <- log_reg_conf_matrix[1, 2]

# Compute evaluation metrics
log_reg_precision <- log_reg_TP / (log_reg_TP + log_reg_FP)
log_reg_recall <- log_reg_TP / (log_reg_TP + log_reg_FN)
log_reg_accuracy <- (log_reg_TP + log_reg_TN) / (log_reg_TP + log_reg_TN + log_reg_FP + log_reg_FN)
log_reg_f1_score <- 2 * (log_reg_precision * log_reg_recall) / (log_reg_precision + log_reg_recall)




# Creating confusion matrix
majority_conf_matrix <- table(
  Predicted = factor(majority_predictions, levels = c(0, 1)),
  Actual    = factor(testing_data_scaled$ICU_outcome, levels = c(0, 1))
)

# Reshape the confusion matrix for ggplot2
conf_matrix_melted <- as.data.frame(majority_conf_matrix)
colnames(conf_matrix_melted) <- c("Actual", "Predicted", "Count")

ggplot(conf_matrix_melted, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "black", size = 6) +  # Add text labels
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted") +
  theme_minimal()

# Extracting evaluation metrics
majority_TP <- majority_conf_matrix[2, 2]
majority_TN <- majority_conf_matrix[1, 1]
majority_FP <- majority_conf_matrix[2, 1]
majority_FN <- majority_conf_matrix[1, 2]

# Compute evaluation metrics
majority_precision <- majority_TP / (majority_TP + majority_FP)
majority_recall <- majority_TP / (majority_TP + majority_FN)
majority_accuracy <- (majority_TP + majority_TN) / (majority_TP + majority_TN + majority_FP + majority_FN)
majority_f1_score <- 2 * (majority_precision * majority_recall) / (majority_precision + majority_recall)


cat("Majority Precision:", majority_precision, "\n")
cat("Majority Recall:", majority_recall, "\n")
cat("Majority Accuracy:", majority_accuracy, "\n")
cat("Majority F1 Score:", majority_f1_score, "\n")

cat("Log Reg Precision:", log_reg_precision, "\n")
cat("Log Reg Recall:", log_reg_recall, "\n")
cat("Log Reg Accuracy:", log_reg_accuracy, "\n")
cat("Log Reg F1 Score:", log_reg_f1_score, "\n")

roc_logreg <- roc(
  response = testing_data_scaled$ICU_outcome,
  predictor = predict_prob,
  levels = c(0, 1),
  direction = "<"
)

auc_logreg <- auc(roc_logreg)
ci_auc <- ci.auc(roc_logreg, method = "delong")

auc_logreg
ci_auc

plot(
  roc_logreg,
  col = "blue",
  lwd = 2,
  main = "ROC curve - Day-1 logistic regression"
)

roc_majority <- roc(
  response = testing_data_scaled$ICU_outcome,
  predictor = majority_predictions,
  levels = c(0, 1),
  direction = "<"
)

auc_majority <- auc(roc_majority)
ci_auc <- ci.auc(roc_majority, method = "delong")

auc_majority
ci_auc

plot(
  roc_majority,
  col = "blue",
  lwd = 2,
  main = "ROC curve - Day-1 logistic regression"
)

roc.test(roc_logreg, roc_majority, method = "delong")