from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from read_data import left_data_original
from Covariance_investigation import X_pca_components_for_model
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

data = left_data_original.copy()
data_day_one = left_data_original[left_data_original["day"] == 1]
seconds_score_day_one = data_day_one["SECONDS_scores"]
max_lag = 0
model_data = pd.DataFrame(X_pca_components_for_model, columns=["PC2", "PC3", "PC4"])

model_data["SECONDS_scores"] = seconds_score_day_one.values
model_data["subject_id"] = data_day_one["subject_id"].values

#for lag in range(1, max_lag+1):

# data[f"LOR_late_gradient_score_lag{lag}"] = data.groupby("subject_id")["LOR_late_gradient_score"].shift(lag)
# data[f"four_score_lag{lag}"] = data.groupby("subject_id")["four_score"].shift(lag)
# data["day_centered"] = data["day"] - data.groupby("subject_id")["day"].transform("min")
# df_lagged = data.dropna(subset=["LOR_late_gradient_score_lag1", "four_score_lag1"])

# model_ar = smf.mixedlm("four_score ~ four_score_lag1 + day_centered",
#                     data=df_lagged,
#                     groups=df_lagged["subject_id"])
# result_ar = model_ar.fit(reml=False)

# model_full = smf.mixedlm("four_score ~ four_score_lag1 + LOR_late_gradient_score_lag1 + day_centered",
#                         data=df_lagged,
#                         groups=df_lagged["subject_id"])
# result_full = model_full.fit(reml=False)

# lr_stat = -2 * (result_ar.llf - result_full.llf)
# p_value = stats.chi2.sf(lr_stat, 1)

# print(f"Lag: {lag}")
# print("result_ar converged:", result_ar.converged)
# print("result_full converged:", result_full.converged)
# print(f"p_value: {np.round(p_value,4)}")
# print(f"LR stat: {np.round(lr_stat, 4)}")

with localconverter(pandas2ri.converter):
        globalenv["model_data"] = model_data   
r('''
library(lme4)
library(lmerTest)
library(performance)
library(see)
library(ggplot2)
library(patchwork)

model_PCA <- glmer(
                SECONDS_scores ~ PC3 + PC4 + (1 | subject_id),
                data=model_data,
                family = binomial(link = "logit"))
model_PCA <- glm(
    SECONDS_scores ~ PC3 + PC4,
    data = model_data,
    family = binomial(link = "logit")
)

# Model summary
print(summary(model_PCA))

# Odds ratios (easier to interpret)
print(exp(coef(model_PCA)))
print(exp(confint(model_PCA)))

# Model fit
library(performance)
print(r2_nagelkerke(model_PCA))
print(AIC(model_PCA))
print(BIC(model_PCA))
''')

# with localconverter(pandas2ri.converter):
#     anova_Condch_df = globalenv["anova_Condch_df"]

# Multilevel logistic regression:
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Fit ordinal logistic regression
model_ordinal = OrderedModel(
    endog=data[data_original["day"] == 1]["FOUR_scores"].values,
    exog=model_data[['PC3', 'PC4']],
    distr='logit'  # or 'probit'
)

result_ordinal = model_ordinal.fit(method='bfgs')
print(result_ordinal.summary())

# Binary logistic regression:
X = model_data[['PC3', 'PC4']]
X = sm.add_constant(X)
y = seconds_score_day_one.values
model_binary = Logit(endog=y, exog=X)
result_binary = model_binary.fit(method='bfgs')
print(result_binary.summary())

print("test")