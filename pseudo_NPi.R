# ============================================================
# NPI reverse engineering — scientific reconstruction (FINAL)
# ============================================================

library(tidyverse)

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
    ch,
    max_const_velocity
  ) %>%
  drop_na()

# ------------------------------------------------------------
# 2. Fixed piecewise-linear rule for pupil_min
# ------------------------------------------------------------

f_pupil_min <- function(pupil_min) {
  
  PM_LOW  <- 4.1
  PM_HIGH <- 6.7
  
  f_LOW  <- -5.2
  f_HIGH <-  0.0
  
  slope_mid <- (f_HIGH - f_LOW) / (PM_HIGH - PM_LOW)
  
  f <- numeric(length(pupil_min))
  
  # poor reflex: saturating penalty
  idx_low <- pupil_min <= PM_LOW
  f[idx_low] <- f_LOW
  
  # transitional: linear ramp
  idx_mid <- pupil_min > PM_LOW & pupil_min < PM_HIGH
  f[idx_mid] <- f_LOW + slope_mid * (pupil_min[idx_mid] - PM_LOW)
  
  # normal: plateau
  idx_high <- pupil_min >= PM_HIGH
  f[idx_high] <- f_HIGH
  
  return(f)
}


# ------------------------------------------------------------
# 3. Reference values for centering (fixed)
# ------------------------------------------------------------

refs <- with(NPi_data, list(
  ps = median(pupil_size),
  v  = median(max_const_velocity),
  ch = median(ch)
))

# ------------------------------------------------------------
# 4. Latent physiology score Q
# ------------------------------------------------------------

Q <- with(
  NPi_data,
  f_pupil_min(pupil_min) +
    0.30 * (pupil_size - refs$ps) +
    0.05 * (max_const_velocity - refs$v) +
    0.02 * (ch - refs$ch)
)

# ------------------------------------------------------------
# 5. Linear mapping to NPI scale
# ------------------------------------------------------------

fit <- lm(NPi ~ Q, data = data.frame(Q = Q, NPi = NPi_data$NPi))
summary(fit)

# ------------------------------------------------------------
# 6. Presentation layer: clip + round
# ------------------------------------------------------------

NPi_hat <- pmin(
  pmax(fit$coefficients[1] + fit$coefficients[2] * Q, 0),
  5
)

NPi_hat_round <- round(NPi_hat, 1)

# ------------------------------------------------------------
# 7. Diagnostics
# ------------------------------------------------------------

summary(NPi_hat_round - NPi_data$NPi)

plot(
  Q,
  NPi_data$NPi,
  pch  = 16,
  col  = rgb(0, 0, 0, 0.3),
  xlab = "Latent physiology score Q",
  ylab = "Observed NPI"
)

plot(
  NPi_hat_round,
  NPi_data$NPi,
  pch  = 16,
  col  = rgb(0, 0, 0, 0.3),
  xlab = "Reconstructed NPI",
  ylab = "Observed NPI"
)
abline(0, 1, col = "red", lwd = 2)
