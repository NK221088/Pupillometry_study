# ============================================================
# NPI reverse engineering — scientific reconstruction
# ============================================================

library(tidyverse)
library(mgcv)
library(gratia)
library(dplyr)

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------

#NPi_measurements <- read_csv(
#  "L:/AuditData/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation/left_NPI_data_cleaned.csv",
#  show_col_types = FALSE
#)

NPi_measurements <- read_csv(
  "C:/Users/NTres/OneDrive - Danmarks Tekniske Universitet/Arbejde_Rigshospitalet/Pupillometry/NPI_investigation/Distribution_investigation/left_NPI_data_cleaned.csv",
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
# 3. Functions to extract smooths
# ------------------------------------------------------------

extract_smooth_df <- function(gam_model, smooth_name, n = 200) {
  
  sm <- smooth_estimates(
    gam_model,
    smooth = smooth_name,
    n = n
  )
  
  sm %>%
    select(x = !!sym(names(sm)[1]), smooth = est)
}


# ------------------------------------------------------------
# 4. Extract all smooths explicitly
# ------------------------------------------------------------

extract_smooth_df <- function(gam_model, smooth_name, n = 200) {
  
  sm <- gratia::smooth_estimates(
    gam_model,
    select = smooth_name,
    n = n
  )
  
  # predictor column = last column (your case)
  x_col <- setdiff(
    names(sm),
    c(".smooth", ".type", ".by", ".estimate", ".se")
  )
  
  if (length(x_col) != 1) {
    stop("Could not uniquely identify predictor column.")
  }
  
  sm[, c(x_col, ".estimate")] |>
    setNames(c("x", "smooth"))
}


# ------------------------------------------------------------
# 5. Construct latent physiology score Q
# ------------------------------------------------------------

smooth_specs <- list(
  pupil_size = list(
    smooth = "s(pupil_size)",
    xlab   = "Pupil size (mm)"
  ),
  pupil_min = list(
    smooth = "s(pupil_min)",
    xlab   = "Minimum pupil size (mm)"
  ),
  max_const_velocity = list(
    smooth = "s(max_const_velocity)",
    xlab   = "Max constriction velocity (mm/s)"
  ),
  ch = list(
    smooth = "s(ch)",
    xlab   = "Constriction amplitude (mm)"
  ),
  dilat_velocity = list(
    smooth = "s(dilat_velocity)",
    xlab   = "Dilatation velocity (mm/s)"
  )
)

out_dir <- "latent_Q_plots"
dir.create(out_dir, showWarnings = FALSE)

for (name in names(smooth_specs)) {
  
  spec <- smooth_specs[[name]]
  
  sm <- extract_smooth_df(
    gam_ref,
    spec$smooth
  )
  
  pdf(
    file = file.path(out_dir, paste0("Q_", name, ".pdf")),
    width = 7,
    height = 5
  )
  
  par(mar = c(4, 4, 2, 1))
  plot(
    sm$x,
    sm$smooth,
    type = "l",
    lwd  = 2,
    xlab = spec$xlab,
    ylab = "Contribution to NPi"
  )
  abline(h = 0, lty = 2)
  
  dev.off()
}

# ------------------------------------------------------------
# 6. Fit pupil size
# ------------------------------------------------------------

eps <- 1e-3  # numerical safety

fit_pup_logpow <- nls(
  smooth ~ alpha * log(x + eps) +
    beta  * pmax(x - x0, 0)^p +
    c0,
  data  = sm_pupil_size,
  start = list(
    alpha = 1,
    beta  = 0.2,
    x0    = x_ref,
    p     = 2,
    c0    = 0
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)
pred_logpow <- predict(fit_pup_logpow)
plot(sm_pupil_size$x, sm_pupil_size$smooth,
     type = "l", lwd = 2,
     xlab = "Pupil size (mm)",
     ylab = "Contribution to NPi")

lines(sm_pupil_size$x, pred_logpow,
      col = "darkgreen", lwd = 2)

abline(v = x_ref, lty = 2)
abline(h = 0, lty = 2)


# ------------------------------------------------------------
# 7. Fit CH
# ------------------------------------------------------------

fit_ch_lm <- lm(smooth ~ poly(x, 2, raw = TRUE), data = sm_ch)

coef(fit_ch_lm)
coefs <- coef(fit_ch_lm)

a <- coefs[3]
b <- coefs[2]
c <- coefs[1]

x0_ch <- -b / (2 * a)
Q0_ch <- a * x0_ch^2 + b * x0_ch + c

x0_ch
Q0_ch
plot(sm_ch$x, sm_ch$smooth, type = "l", lwd = 2)
lines(sm_ch$x,
      a * sm_ch$x^2 + b * sm_ch$x + c,
      col = "red", lwd = 2)
abline(v = x0_ch, lty = 2)

# ------------------------------------------------------------
# 8. Fit Min. pupil size
# ------------------------------------------------------------

min_row <- sm_pupil_min[which.min(sm_pupil_min$smooth), ]

x0_min <- min_row$x
c0_min <- min_row$smooth

x0_min
c0_min

Q_pupil_min <- function(x, a, x0 = x0_min, c0 = c0_min) {
  a * (x - x0)^2 + c0
}

fit_min <- nls(
  smooth ~ Q_pupil_min(x, a),
  data  = sm_pupil_min,
  start = list(a = 0.1)
)

coef(fit_min)

plot(sm_pupil_min$x, sm_pupil_min$smooth,
     type = "l", lwd = 2,
     xlab = "Minimum pupil size (mm)",
     ylab = "Contribution to NPi")

lines(sm_pupil_min$x,
      Q_pupil_min(sm_pupil_min$x, coef(fit_min)["a"]),
      col = "red", lwd = 2)

abline(v = x0_min, lty = 2)

Q_min_pupil_size <- function(x, a) {
  a * (x - x0_min)^2 + c0_min
}

# ------------------------------------------------------------
# 9. Fit reconstructed NPi (NPi_hat) using core 3 terms
# ------------------------------------------------------------

# ---- 9.1 Define fitted Q-functions using learned parameters ----

# Pupil size term (log + power)
coef_pup <- coef(fit_pup_logpow)

Q_pupil_size <- function(x) {
  coef_pup["alpha"] * log(x + eps) +
    coef_pup["beta"] * pmax(x - coef_pup["x0"], 0)^coef_pup["p"] +
    coef_pup["c0"]
}

# CH term (concave quadratic)
Q_ch <- function(x) {
  a * (x - x0_ch)^2 + Q0_ch
}

# Min pupil size term (penalty quadratic)
coef_min <- coef(fit_min)

Q_pupil_min <- function(x) {
  coef_min["a"] * (x - x0_min)^2 + c0_min
}

# ---- 9.2 Fit weights + intercept (EXPLICIT parameters) ----

fit_npi_hat <- nls(
  NPi ~ beta0 +
    w_pupil * Q_pupil_size(pupil_size) +
    w_min   * Q_pupil_min(pupil_min) +
    w_ch    * Q_ch(ch),
  data  = NPi_gam,
  start = list(
    beta0  = mean(NPi_gam$NPi),
    w_pupil = 1,
    w_min   = 1,
    w_ch    = 1
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)

coef(fit_npi_hat)

NPi_gam$NPi_hat <- predict(fit_npi_hat)

# RMSE
rmse_npi_hat <- sqrt(mean((NPi_gam$NPi_hat - NPi_gam$NPi)^2))
rmse_npi_hat

# Observed vs reconstructed
plot(NPi_gam$NPi, NPi_gam$NPi_hat,
     xlab = "Observed NPi",
     ylab = "Reconstructed NPi",
     pch  = 16, col = rgb(0,0,0,0.4))
abline(0, 1, col = "red", lwd = 2)

# Residuals
plot(NPi_gam$NPi_hat,
     NPi_gam$NPi - NPi_gam$NPi_hat,
     xlab = "Reconstructed NPi",
     ylab = "Residual",
     pch  = 16, col = rgb(0,0,0,0.4))
abline(h = 0, lty = 2)

SSE <- sum((NPi_gam$NPi - NPi_gam$NPi_hat)^2)
SST <- sum((NPi_gam$NPi - mean(NPi_gam$NPi))^2)

R2 <- 1 - SSE / SST
R2

n <- nrow(NPi_gam)
k <- length(coef(fit_npi_hat))  # number of fitted parameters

R2_adj <- 1 - (1 - R2) * (n - 1) / (n - k - 1)
R2_adj

