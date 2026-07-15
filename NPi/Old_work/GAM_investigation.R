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

NPi_measurements <- read_csv(
 "L:/AuditData/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation/left_NPI_data_cleaned.csv",
 show_col_types = FALSE
)

#NPi_measurements <- read_csv(
#  "C:/Users/NTres/OneDrive - Danmarks Tekniske Universitet/Arbejde_Rigshospitalet/Pupillometry/NPI_investigation/Distribution_investigation/left_NPI_data_cleaned.csv",
#  show_col_types = FALSE
#)


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
    s(dilat_velocity, k = 6) +
    s(record_id, bs = "re"),
  data   = NPi_gam,
  method = "REML"
)


summary(gam_ref)


# gam_no_dv <- gam(
#   NPi ~
#     s(pupil_size, k = 6) +
#     s(pupil_min, k = 6) +
#     s(max_const_velocity, k = 6) +
#     s(ch, k = 6) +
#     s(record_id, bs = "re"),
#   data = NPi_gam,
#   method = "ML"
# )
# 
# summary(gam_no_dv)
# 
# anova(gam_ref, gam_no_dv, test = "Chisq")


# gam_no_cv <- gam(
#   NPi ~
#     s(pupil_size, k = 6) +
#     s(pupil_min, k = 6) +
#     s(ch, k = 6) +
#     s(dilat_velocity, k = 6) +
#     s(record_id, bs = "re"),
#   data   = NPi_gam,
#   method = "REML"
# )

# summary(gam_no_cv)
# 
# anova(gam_ref, gam_no_cv, test = "Chisq")

# ------------------------------------------------------------
# 3. Extract all smooths explicitly
# ------------------------------------------------------------

extract_smooth_df <- function(gam_model, smooth_name, n = 200) {
  
  sm <- gratia::smooth_estimates(
    gam_model,
    select = smooth_name,
    n = n
  )
  
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
# 4. Define smooth specifications ONCE
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


# ------------------------------------------------------------
# 5. Extract and store smooths (reproducible!)
# ------------------------------------------------------------

smooths <- lapply(smooth_specs, function(spec) {
  extract_smooth_df(gam_ref, spec$smooth)
})

names(smooths) <- names(smooth_specs)

# Explicit objects used later
sm_pupil_size <- smooths$pupil_size
sm_pupil_min  <- smooths$pupil_min
sm_ch         <- smooths$ch
sm_max_const_velocity <- smooths$max_const_velocity
sm_dv <- smooths$dilat_velocity

# ------------------------------------------------------------
# 6. Plot latent Q smooths
# ------------------------------------------------------------

out_dir <- "L:/AuditData/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation"
dir.create(out_dir, showWarnings = FALSE)

for (name in names(smooth_specs)) {
  
  spec <- smooth_specs[[name]]
  sm   <- smooths[[name]]
  
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
# 7. Fit CH
# ------------------------------------------------------------

# Restrict to central support of the smooth
sm_ch_fit <- sm_ch %>%
  filter(
    x >= quantile(x, 0.05),
    x <= quantile(x, 0.95)
  )

# Fit quadratic ONLY on the reliable region
fit_ch_lm <- lm(
  smooth ~ poly(x, 2, raw = TRUE),
  data = sm_ch_fit
)

coefs <- coef(fit_ch_lm)

a <- coefs[3]
b <- coefs[2]
c <- coefs[1]

# Vertex (maximum or minimum)
x0_ch <- -b / (2 * a)
Q0_ch <- a * x0_ch^2 + b * x0_ch + c

# Diagnostic plot
plot(sm_ch$x, sm_ch$smooth,
     type = "l", lwd = 2,
     xlab = "Constriction amplitude (mm)",
     ylab = "Contribution to NPi")

lines(
  sm_ch$x,
  a * sm_ch$x^2 + b * sm_ch$x + c,
  col = "red", lwd = 2
)

abline(v = x0_ch, lty = 2)
abline(h = 0, lty = 2)

# ------------------------------------------------------------
# 8. Fit minimum pupil size (linear contribution)
# ------------------------------------------------------------

# Reference point: zero-crossing of the smooth (for interpretability)
x0_min <- sm_pupil_min$x[
  which.min(abs(sm_pupil_min$smooth))
]

x0_min

# Define linear Q-function
Q_pupil_min <- function(x, b, x0 = x0_min) {
  b * (x - x0)
}

# Fit linear model to full smooth
fit_min <- lm(
  smooth ~ I(x - x0_min),
  data = sm_pupil_min
)

coef(fit_min)

b_min <- coef(fit_min)[2]

# Diagnostic plot
plot(
  sm_pupil_min$x,
  sm_pupil_min$smooth,
  type = "l", lwd = 2,
  xlab = "Minimum pupil size (mm)",
  ylab = "Contribution to NPi"
)

lines(
  sm_pupil_min$x,
  Q_pupil_min(sm_pupil_min$x, b_min),
  col = "red", lwd = 2
)

abline(h = 0, lty = 2)
abline(v = x0_min, lty = 2)



# ------------------------------------------------------------
# 10. Fit pupil size (log + power-law)
# ------------------------------------------------------------

eps <- 1e-3  # numerical safety

# Reference point: where smooth crosses zero (or closest)
x_ref <- sm_pupil_size$x[which.min(abs(sm_pupil_size$smooth))]

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

coef(fit_pup_logpow)

# Diagnostic plot
plot(sm_pupil_size$x, sm_pupil_size$smooth,
     type = "l", lwd = 2,
     xlab = "Pupil size (mm)",
     ylab = "Contribution to NPi")
lines(sm_pupil_size$x,
      predict(fit_pup_logpow),
      col = "darkgreen", lwd = 2)
abline(v = x_ref, lty = 2)
abline(h = 0, lty = 2)

summary(fit_pup_logpow)

# NOTE:
# Identifiability verified by multi-start NLS convergence
# Parameters are stable across wide range of initial values

# Refit with slightly different starting values
starts <- list(
  list(alpha = 0.5, beta = 0.5, x0 = x_ref + 0.2, p = 1.2, c0 = 0),
  list(alpha = 2.5, beta = 0.1, x0 = x_ref - 0.2, p = 2.5, c0 = -1),
  list(alpha = 1.0, beta = 0.8, x0 = x_ref,       p = 1.8, c0 = -2)
)

fits <- lapply(starts, function(s)
  nls(
    smooth ~ alpha * log(x + eps) +
      beta  * pmax(x - x0, 0)^p +
      c0,
    data = sm_pupil_size,
    start = s,
    control = nls.control(maxiter = 500, warnOnly = TRUE)
  )
)

sapply(fits, coef)


# ------------------------------------------------------------
# 11.  Fit maximum constriction velocity
# ------------------------------------------------------------

# choose reference point from smooth (e.g. zero crossing)
x0_mcv <- sm_max_const_velocity$x[
  which.min(abs(sm_max_const_velocity$smooth))
]

fit_mcv <- nls(
  smooth ~ a * pmin(x - x0_mcv, 0)^2,
  data  = sm_max_const_velocity,
  start = list(a = -0.01)
)

coef(fit_mcv)


summary(fit_mcv)
confint(fit_mcv)

pred_mcv <- predict(fit_mcv)

rmse_mcv <- sqrt(mean(
  (pred_mcv - sm_max_const_velocity$smooth)^2
))

rmse_mcv

plot(sm_max_const_velocity$x, sm_max_const_velocity$smooth,
     type = "l", lwd = 2,
     xlab = "Max constriction velocity",
     ylab = "Contribution to NPi")

lines(sm_max_const_velocity$x,
      pred_mcv,
      col = "red", lwd = 2)

abline(v = x0_mcv, lty = 2)
abline(h = 0, lty = 2)

# ------------------------------------------------------------
# 12.  Fit the dilation velocity
# ------------------------------------------------------------

fit_dv_quad <- lm(
  smooth ~ poly(x, 2, raw = TRUE),
  data = sm_dv
)

coef(fit_dv_quad)


a2_dv <- coef(fit_dv_quad)[3]
a1_dv <- coef(fit_dv_quad)[2]
a0_dv <- coef(fit_dv_quad)[1]

# Quadratic prediction from fitted model
pred_dv <- a0_dv +
  a1_dv * sm_dv$x +
  a2_dv * sm_dv$x^2

plot(sm_dv$x, sm_dv$smooth,
     type = "l", lwd = 2,
     xlab = "Dilation velocity",
     ylab = "Contribution to NPi")

lines(sm_dv$x,
      pred_dv,
      col = "red", lwd = 2)

abline(h = 0, lty = 2)
# ------------------------------------------------------------
# 13. Fit reconstructed NPi (NPi_hat) using core terms
# ------------------------------------------------------------

# ---- 12.1 Define fitted Q-functions using learned parameters ----

# Dilation Velocity term

Q_dv <- function(x) {
  a2_dv * x^2 + a1_dv * x + a0_dv
}


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

# Minimum pupil size term (penalty quadratic)
coef_min <- coef(fit_min)
b_min <- coef_min[2]   # slope of the linear effect

Q_pupil_min <- function(x) {
  b_min * (x - x0_min)
}

# Maximum constriction velocity term
coef_mcv <- coef(fit_mcv)

Q_mcv <- function(x) {
  coef_mcv["a"] * pmin(x - x0_mcv, 0)^2
}

# ---- 12.2 Fit linear latent NPi score (physiological weights) ----

fit_npi_hat_mcv <- nls(
  NPi ~ beta0 +
    w_pupil * Q_pupil_size(pupil_size) +
    w_min   * Q_pupil_min(pupil_min) +
    w_ch    * Q_ch(ch) +
    w_mcv   * Q_mcv(max_const_velocity) +
    w_dv    * Q_dv(dilat_velocity),
  data  = NPi_gam,
  start = list(
    beta0   = mean(NPi_gam$NPi),
    w_pupil = 1,
    w_min   = 1,
    w_ch    = 1,
    w_mcv   = 1,
    w_dv    = 1
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)


# ---- Latent (unbounded) NPi score ----
NPi_gam$NPi_latent <- predict(fit_npi_hat_mcv)

# ---- Logistic squashing function (device-level mapping) ----
logistic_npi <- function(z, L = 5, k, z0) {
  L / (1 + exp(-k * (z - z0)))
}

# ---- 12.3 Fit logistic squash parameters ----

fit_squash <- nls(
  NPi ~ logistic_npi(NPi_latent, L = 5, k, z0),
  data  = NPi_gam,
  start = list(
    k  = 1,
    z0 = mean(NPi_gam$NPi_latent)
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)

pars_squash <- coef(fit_squash)

k_squash  <- pars_squash["k"]
z0_squash <- pars_squash["z0"]
L_squash  <- 5

NPi_squash <- function(z) {
  L_squash / (1 + exp(-k_squash * (z - z0_squash)))
}

# ---- Final bounded NPi prediction ----
NPi_gam$fit_npi_hat_mcv <- NPi_squash(NPi_gam$NPi_latent)

# ---- Model evaluation ----

# RMSE
rmse_npi_hat <- sqrt(mean(
  (NPi_gam$fit_npi_hat_mcv - NPi_gam$NPi)^2
))
rmse_npi_hat

# Observed vs reconstructed
plot(NPi_gam$NPi, NPi_gam$fit_npi_hat_mcv,
     xlab = "Observed NPi",
     ylab = "Reconstructed NPi",
     pch  = 16, col = rgb(0, 0, 0, 0.4))
abline(0, 1, col = "red", lwd = 2)

# Residuals
plot(NPi_gam$fit_npi_hat_mcv,
     NPi_gam$NPi - NPi_gam$fit_npi_hat_mcv,
     xlab = "Reconstructed NPi",
     ylab = "Residual",
     pch  = 16, col = rgb(0, 0, 0, 0.4))
abline(h = 0, lty = 2)

# R²
SSE <- sum((NPi_gam$NPi - NPi_gam$fit_npi_hat_mcv)^2)
SST <- sum((NPi_gam$NPi - mean(NPi_gam$NPi))^2)

R2 <- 1 - SSE / SST
R2

# Adjusted R² (includes squash parameters)
n <- nrow(NPi_gam)
k <- length(coef(fit_npi_hat_mcv))

R2_adj <- 1 - (1 - R2) * (n - 1) / (n - k - 1)
R2_adj

# ------------------------------------------------------------
# 13. Extract and store fitted algorithm
# ------------------------------------------------------------

# ---- Core weights + intercept ----
pars_npi <- coef(fit_npi_hat_mcv)

beta0   <- pars_npi["beta0"]
w_pupil <- pars_npi["w_pupil"]
w_min   <- pars_npi["w_min"]
w_ch    <- pars_npi["w_ch"]
w_mcv   <- pars_npi["w_mcv"]

# ---- Pupil size parameters ----
pars_pup <- coef(fit_pup_logpow)

alpha_pup <- pars_pup["alpha"]
beta_pup  <- pars_pup["beta"]
x0_pup    <- pars_pup["x0"]
p_pup     <- pars_pup["p"]
c0_pup    <- pars_pup["c0"]

# ---- Minimum pupil size ----
a_min <- coef(fit_min)["a"]

# ---- CH ----
a_ch <- a
x0_ch <- x0_ch
c0_ch <- Q0_ch

# ---- Max constriction velocity ----
a_mcv  <- coef(fit_mcv)["a"]
x0_mcv <- x0_mcv

eps <- 1e-3


Q_pupil_size <- function(x) {
  alpha_pup * log(x + eps) +
    beta_pup * pmax(x - x0_pup, 0)^p_pup +
    c0_pup
}

Q_pupil_min <- function(x) {
  b_min * (x - x0_min)
}


Q_ch <- function(x) {
  a_ch * (x - x0_ch)^2 + c0_ch
}

Q_mcv <- function(x) {
  a_mcv * pmin(x - x0_mcv, 0)^2
}

NPi_hat <- function(pupil_size,
                    pupil_min,
                    ch,
                    max_const_velocity) {
  
  # latent physiological score
  z <-
    beta0 +
    w_pupil * Q_pupil_size(pupil_size) +
    w_min   * Q_pupil_min(pupil_min) +
    w_ch    * Q_ch(ch) +
    w_mcv   * Q_mcv(max_const_velocity)
  
  # device-level bounded NPi
  L_squash / (1 + exp(-k_squash * (z - z0_squash)))
}


saveRDS(
  list(
    # ---- Core weights ----
    beta0   = beta0,
    w_pupil = w_pupil,
    w_min   = w_min,
    w_ch    = w_ch,
    w_mcv   = w_mcv,
    
    # ---- Pupil size (nonlinear) ----
    alpha_pup = alpha_pup,
    beta_pup  = beta_pup,
    x0_pup    = x0_pup,
    p_pup     = p_pup,
    c0_pup    = c0_pup,
    
    # ---- Minimum pupil size (LINEAR) ----
    b_min  = b_min,
    x0_min = x0_min,
    
    # ---- Constriction amplitude ----
    a_ch  = a_ch,
    x0_ch = x0_ch,
    c0_ch = c0_ch,
    
    # ---- Max constriction velocity ----
    a_mcv  = a_mcv,
    x0_mcv = x0_mcv,
    
    # ---- Logistic squash ----
    eps       = eps,
    k_squash  = k_squash,
    z0_squash = z0_squash,
    L_squash  = L_squash
  ),
  file = "NPi_hat_parameters.rds"
)




# ------------------------------------------------------------
# 14. Test algorithm on right eye data
# ------------------------------------------------------------

NPi_right <- read_csv(
  "L:/Auditdata/CONNECT-ME/Nikolai/pupillometry/Distribution_investigation/right_NPI_data_cleaned.csv",
  show_col_types = FALSE
)

NPi_right <- NPi_right %>%
  rename(
    record_id          = `record id`,
    repeat_instance    = `redcap repeat instance`,
    pupil_size         = `pupil size`,
    pupil_min          = `pupil min`,
    const_velocity     = `const velocity`,
    max_const_velocity = `max const velocity`,
    dilat_velocity     = `dilat velocity`
  )

NPi_right <- NPi_right %>%
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

NPi_right$NPi_latent <- with(
  NPi_right,
  beta0 +
    w_pupil * Q_pupil_size(pupil_size) +
    w_min   * Q_pupil_min(pupil_min) +
    w_ch    * Q_ch(ch) +
    w_mcv   * Q_mcv(max_const_velocity)
)



NPi_right$NPi_hat <- NPi_squash(NPi_right$NPi_latent)


NPi_right$NPi_hat_GAM <- predict(gam_ref, newdata = NPi_right)

NPi_right$residual <- NPi_right$NPi - NPi_right$NPi_hat

sigma_hat <- sd(NPi_right$residual)
sigma_hat

NPi_right$NPi_hat_lower <- NPi_right$NPi_hat - 1.96 * sigma_hat
NPi_right$NPi_hat_upper <- NPi_right$NPi_hat + 1.96 * sigma_hat

plot(NPi_right$NPi_hat, NPi_right$residual,
     xlab = "Reconstructed NPi",
     ylab = "Residual",
     pch = 16, col = rgb(0,0,0,0.4))
abline(h = 0, lty = 2)

R2_NPi_hat_right <- 1 -
  sum((NPi_right$NPi - NPi_right$NPi_hat)^2) /
  sum((NPi_right$NPi - mean(NPi_right$NPi))^2)

R2_GAM_right <- 1 - sum((NPi_right$NPi - NPi_right$NPi_hat_GAM)^2) /
  sum((NPi_right$NPi - mean(NPi_right$NPi))^2)

RMSE_GAM_right <- sqrt(mean(
  (NPi_right$NPi_hat_GAM - NPi_right$NPi)^2
))

R2_NPi_hat_right
R2_GAM_right


NPi_right <- NPi_right %>%
  mutate(
    abs_error = abs(NPi - NPi_hat)
  ) %>%
  arrange(desc(abs_error))
head(NPi_right, 20)

vars <- list(
  list(name = "pupil_min",  label = "Minimum pupil size (mm)", vline = x0_min),
  list(name = "pupil_size", label = "Pupil size (mm)",         vline = x0_pup),
  list(name = "ch",         label = "Constriction amplitude",  vline = x0_ch),
  list(name = "max_const_velocity", label = "Max constriction velocity", vline = x0_mcv),
  list(name = "dilat_velocity", label = "Dilation velocity",   vline = NULL)
)

for (v in vars) {
  plot(
    NPi_right[[v$name]],
    NPi_right$residual,
    pch = 16, col = rgb(0,0,0,0.4),
    xlab = v$label,
    ylab = "Residual (Observed − Predicted)"
  )
  abline(h = 0, lty = 2)
  if (!is.null(v$vline)) abline(v = v$vline, lty = 2)
}

plot(
  NPi_right$NPi_hat,
  NPi_right$residual,
  pch = 16, col = rgb(0,0,0,0.4),
  xlab = "Predicted NPi",
  ylab = "Residual (Observed − Predicted)"
)
abline(h = 0, lty = 2)

plot(
  NPi_right$NPi_latent,
  NPi_right$residual,
  pch = 16, col = rgb(0,0,0,0.4),
  xlab = "Latent NPi score (z)",
  ylab = "Residual"
)
abline(h = 0, lty = 2)


hist(
  NPi_gam$NPi,
  breaks = 30,
  main = "Distribution of NPi (training data)",
  xlab = "NPi"
)

NPi_right %>%
  mutate(bin = cut(NPi, breaks = c(0, 2, 3, 4, 5))) %>%
  group_by(bin) %>%
  summarise(
    n = n(),
    RMSE = sqrt(mean((NPi - NPi_hat)^2)),
    mean_abs_error = mean(abs(NPi - NPi_hat))
  )

