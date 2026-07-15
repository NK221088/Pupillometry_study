# ============================================================
# NPI reverse engineering — scientific reconstruction (FIRST DRAFT)
# ============================================================

library(tidyverse)
library(dplyr)

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------

NPi_measurements <- read_csv(
  "L:/AuditData/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation/left_NPI_data_cleaned.csv",
  show_col_types = FALSE
)

NPi_HC_left_measurements <- read_csv(
  "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/HC_left_NPi.csv",
  show_col_types = FALSE
) %>%
  rename_with(~ str_remove(.x, "_left$"))

NPi_HC_right_measurements <- read_csv(
  "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Data/HC_right_NPi.csv",
  show_col_types = FALSE
) %>%
  rename_with(~ str_remove(.x, "_right$"))

NPi_HC_measurements <- bind_rows(
  NPi_HC_left_measurements,
  NPi_HC_right_measurements
) %>%
  rename(
    NPi = npi
  )

NPi_measurements <- NPi_measurements %>%
  rename(
    record_id          = `record id`,
    repeat_instance    = `redcap repeat instance`,
    pupil_size         = `pupil size`,
    pupil_min          = `pupil min`,
    ch                 = ch,
    const_velocity     = `const velocity`,
    max_const_velocity = `max const velocity`,
    latency            = latency,
    dilat_velocity     = `dilat velocity`
  )

NPi_data <- NPi_measurements %>%
  select(
    NPi,
    pupil_size,
    pupil_min,
    ch,
    const_velocity,
    max_const_velocity,
    latency,
    dilat_velocity
  ) %>%
  drop_na()

# ------------------------------------------------------------
# 2. Literature-based reference values
# ------------------------------------------------------------

norm_ref <- list(
  pupil_size         = c(mean = 4.31, sd = 0.27),
  pupil_min          = c(mean = 2.83, sd = 0.45),
  ch                 = c(mean = 32.36, sd = 7.90),
  const_velocity     = c(mean = 2.82, sd = 1.11),
  max_const_velocity = c(mean = 3.99, sd = 1.14),
  latency            = c(mean = 0.22, sd = 0.04),
  dilat_velocity     = c(mean = 1.21, sd = 0.33)
)

NPi_data_Z_literature <- NPi_data %>%
  mutate(
    z_pupil_size         = (pupil_size - norm_ref$pupil_size["mean"]) / norm_ref$pupil_size["sd"],
    z_pupil_min          = (pupil_min - norm_ref$pupil_min["mean"]) / norm_ref$pupil_min["sd"],
    z_ch                 = (ch - norm_ref$ch["mean"]) / norm_ref$ch["sd"],
    z_const_velocity     = (const_velocity - norm_ref$const_velocity["mean"]) / norm_ref$const_velocity["sd"],
    z_max_const_velocity = (max_const_velocity - norm_ref$max_const_velocity["mean"]) / norm_ref$max_const_velocity["sd"],
    z_latency            = (latency - norm_ref$latency["mean"]) / norm_ref$latency["sd"],
    z_dilat_velocity     = (dilat_velocity - norm_ref$dilat_velocity["mean"]) / norm_ref$dilat_velocity["sd"]
  )

# ------------------------------------------------------------
# 3. Helper functions for local healthy-control normalization
# ------------------------------------------------------------

build_normative_model <- function(data, vars) {
  X <- data[, vars, drop = FALSE]
  
  mu <- colMeans(X, na.rm = TRUE)
  sds <- apply(X, 2, sd, na.rm = TRUE)
  R <- cor(X, use = "pairwise.complete.obs")
  
  list(
    vars = vars,
    mu = mu,
    sds = sds,
    R = R,
    n = nrow(X)
  )
}

compute_z_scores <- function(data, norm_model) {
  Z <- sweep(data[, norm_model$vars, drop = FALSE], 2, norm_model$mu, "-")
  Z <- sweep(Z, 2, norm_model$sds, "/")
  as.data.frame(Z)
}

evaluate_continuous <- function(obs, pred) {
  rmse <- sqrt(mean((obs - pred)^2))
  mae  <- mean(abs(obs - pred))
  r    <- cor(obs, pred)
  r2   <- 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  
  data.frame(
    RMSE = rmse,
    MAE = mae,
    Correlation = r,
    R2 = r2
  )
}

# ------------------------------------------------------------
# 4. Build local healthy-control standardization
# ------------------------------------------------------------

vars <- c(
  "pupil_size",
  "pupil_min",
  "ch",
  "const_velocity",
  "max_const_velocity",
  "latency",
  "dilat_velocity"
)

norm_model <- build_normative_model(NPi_HC_measurements, vars)

Z_local <- compute_z_scores(NPi_data, norm_model) %>%
  rename(
    z_pupil_size         = pupil_size,
    z_pupil_min          = pupil_min,
    z_ch                 = ch,
    z_const_velocity     = const_velocity,
    z_max_const_velocity = max_const_velocity,
    z_latency            = latency,
    z_dilat_velocity     = dilat_velocity
  )

NPi_data_Z_local <- bind_cols(
  NPi = NPi_data$NPi,
  Z_local
)

# ------------------------------------------------------------
# 5. Choose which standardization to use for modelling
# ------------------------------------------------------------

# Set to TRUE once the healthy control sample is considered stable enough
use_local_normative_model <- FALSE

if (use_local_normative_model) {
  Z <- NPi_data_Z_local
} else {
  Z <- NPi_data_Z_literature
}

# ------------------------------------------------------------
# 6. Train/test split
# ------------------------------------------------------------

set.seed(42)

train_idx <- sample(seq_len(nrow(Z)), size = 0.8 * nrow(Z))

train <- Z[train_idx, ]
test  <- Z[-train_idx, ]

# ------------------------------------------------------------
# 7. Fit models
# ------------------------------------------------------------

fit_linear <- lm(
  NPi ~ z_pupil_size +
    z_pupil_min +
    z_ch +
    z_const_velocity +
    z_max_const_velocity +
    z_latency +
    z_dilat_velocity,
  data = train
)

fit_interaction <- lm(
  NPi ~ (z_pupil_size +
           z_pupil_min +
           z_ch +
           z_const_velocity +
           z_max_const_velocity +
           z_latency +
           z_dilat_velocity)^2,
  data = train
)

# ------------------------------------------------------------
# 8. Evaluate models
# ------------------------------------------------------------

summary(fit_linear)
summary(fit_interaction)

test$pred_linear <- predict(fit_linear, newdata = test)
test$pred_linear <- pmin(pmax(test$pred_linear, 0), 5)

test$pred_interaction <- predict(fit_interaction, newdata = test)
test$pred_interaction <- pmin(pmax(test$pred_interaction, 0), 5)

evaluate_continuous(test$NPi, test$pred_linear)
evaluate_continuous(test$NPi, test$pred_interaction)

ggplot(test, aes(x = NPi, y = pred_linear)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed")

ggplot(test, aes(x = NPi, y = pred_interaction)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed")