# ============================================================
# NPI reverse engineering — scientific reconstruction (FINAL)
# ============================================================

library(tidyverse)
library(dplyr)
library(ordinal)

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------

NPi_measurements <- read_csv(
  "L:/AuditData/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation/left_NPI_data_cleaned.csv",
  show_col_types = FALSE
)

NPi_measurements <- NPi_measurements %>%
  rename(
    record_id          = `record id`,
    repeat_instance    = `redcap repeat instance`,
    pupil_size         = `pupil size`,
    pupil_min          = `pupil min`,
    const_velocity     = `const velocity`,
    max_const_velocity = `max const velocity`,
    dilat_velocity     = `dilat velocity`
  )

NPi_data <- NPi_measurements %>%
  select(
    NPi,
    pupil_size,
    pupil_min,
    const_velocity,
    max_const_velocity,
    dilat_velocity
  ) %>%
  drop_na()

norm_ref <- list(
  pupil_size = c(mean = 4.31, sd = 0.27),
  pupil_min = c(mean = 2.83, sd = 0.45),
  const_velocity = c(mean = 2.82, sd = 1.11),
  max_const_velocity = c(mean = 3.99, sd = 1.14),
  dilat_velocity = c(mean = 1.21, sd = 0.33)
)

NPi_data_Z <- NPi_data %>%
  mutate(
    z_pupil_size         = (pupil_size - norm_ref$pupil_size["mean"]) / norm_ref$pupil_size["sd"],
    z_pupil_min          = (pupil_min - norm_ref$pupil_min["mean"]) / norm_ref$pupil_min["sd"],
    z_const_velocity     = (const_velocity - norm_ref$const_velocity["mean"]) / norm_ref$const_velocity["sd"],
    z_max_const_velocity = (max_const_velocity - norm_ref$max_const_velocity["mean"]) / norm_ref$max_const_velocity["sd"],
    z_dilat_velocity     = (dilat_velocity - norm_ref$dilat_velocity["mean"]) / norm_ref$dilat_velocity["sd"]
  )

fit_lm <- lm(
  NPi ~ z_pupil_size + z_pupil_min + z_const_velocity +
    z_max_const_velocity + z_dilat_velocity,
  data = NPi_data_Z
)

summary(fit_lm)

logistic_5 <- function(s, k) {
  5 / (1 + exp(-k * s))
}

fit_nls <- nls(
  NPi ~ logistic_5(
    beta0 +
      w1 * z_pupil_size +
      w2 * z_pupil_min +
      w3 * z_const_velocity +
      w4 * z_max_const_velocity +
      w5 * z_dilat_velocity,
    k
  ),
  data = NPi_data_Z,
  start = list(
    beta0 = unname(coef(fit_lm)[1]),
    w1    = unname(coef(fit_lm)[2]),
    w2    = unname(coef(fit_lm)[3]),
    w3    = unname(coef(fit_lm)[4]),
    w4    = unname(coef(fit_lm)[5]),
    w5    = unname(coef(fit_lm)[6]),
    k     = 1
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)