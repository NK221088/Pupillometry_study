# ============================================================
# NPI reverse engineering — scientific reconstruction
# ============================================================

library(tidyverse)
library(mgcv)
library(gratia)

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

NPi_gam <- NPi_measurements %>%
  select(
    NPi,
    record_id,
    pupil_size,
    pupil_min,
    ch,
    const_velocity,
    max_const_velocity,
    latency,
    dilat_velocity
  ) %>%
  drop_na() %>%
  mutate(record_id = factor(record_id))

# ------------------------------------------------------------
# 2. Discovery GAM (used only to learn structure)
# ------------------------------------------------------------

gam_ref <- gam(
  NPi ~
    s(pupil_size, k = 6) +
    s(pupil_min, k = 6) +
    s(max_const_velocity, k = 6) +
    s(ch, k = 6) +
    s(dilat_velocity, k = 6),
  data   = NPi_gam,
  method = "REML"
)

summary(gam_ref)

# ------------------------------------------------------------
# 3. Extract smooth for pupil_min (dominant nonlinear term)
# ------------------------------------------------------------

sm <- smooth_estimates(gam_ref)

pmin_sm <- sm %>%
  filter(.smooth == "s(pupil_min)") %>%
  select(pupil_min, f_pmin = .estimate)

# Explicit interpolated function (temporary; will be replaced later)
f_pupil_min <- function(x) {
  approx(
    x     = pmin_sm$pupil_min,
    y     = pmin_sm$f_pmin,
    xout  = x,
    rule  = 2
  )$y
}

# ------------------------------------------------------------
# 4. Define reference values for centering
# ------------------------------------------------------------

refs <- with(NPi_gam, list(
  ps = median(pupil_size),
  v  = median(max_const_velocity),
  ch = median(ch)
))

# ------------------------------------------------------------
# 5. Construct latent physiology score Q
# ------------------------------------------------------------

Q <- with(
  NPi_gam,
  f_pupil_min(pupil_min) +
    0.30 * (pupil_size - refs$ps) +
    0.05 * (max_const_velocity - refs$v) +
    0.02 * (ch - refs$ch)
)

# Diagnostic plot: latent score vs observed NPI
plot(
  Q,
  NPi_gam$NPi,
  pch  = 16,
  col  = rgb(0, 0, 0, 0.3),
  xlab = "Latent physiology score Q",
  ylab = "Observed NPI"
)

# ------------------------------------------------------------
# 6. Linear mapping from Q to NPI scale
# ------------------------------------------------------------

fit <- lm(NPi ~ Q, data = data.frame(Q = Q, NPi = NPi_gam$NPi))
summary(fit)

# ------------------------------------------------------------
# 7. Apply clipping and rounding (presentation layer)
# ------------------------------------------------------------

NPi_hat <- pmin(
  pmax(fit$coefficients[1] + fit$coefficients[2] * Q, 0),
  5
)

NPi_hat_round <- round(NPi_hat, 1)

# ------------------------------------------------------------
# 8. Final diagnostics
# ------------------------------------------------------------

summary(NPi_hat_round - NPi_gam$NPi)

plot(
  NPi_hat_round,
  NPi_gam$NPi,
  pch  = 16,
  col  = rgb(0, 0, 0, 0.3),
  xlab = "Reconstructed NPI",
  ylab = "Observed NPI"
)
abline(0, 1, col = "red", lwd = 2)
