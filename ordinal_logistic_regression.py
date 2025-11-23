from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import Logit
from read_data import left_data_original
import numpy as np

# Create the pairplot
first_day_data = left_data_original[left_data_original["Day"] == 1]
diagnostic_metric = "FOUR"
diagnostic_metric = f"{diagnostic_metric} scores"
predictors = list(first_day_data.columns[5:])
chosen_predictors = {}
y = first_day_data[diagnostic_metric]
run_count = 1
while len(predictors) > 0:
    results = []
    current_predictors = chosen_predictors[np.max(list(chosen_predictors.keys()))][0] if len(list(chosen_predictors.keys())) > 0 else []
    for predictor in predictors:
        if diagnostic_metric == "SECONDS scores":
            model = Logit(
                                endog=y,
                                exog=first_day_data[current_predictors + [predictor]])
        elif diagnostic_metric == "FOUR scores":
            model = OrderedModel(
                    endog=y,
                    exog=first_day_data[current_predictors + [predictor]],
                    distr="logit"
                    )
        model_results = model.fit(method='bfgs')
        results.append(model_results.bic)
    current_min = chosen_predictors[list(np.max(chosen_predictors.keys()))[0]][1] if len(list(chosen_predictors.keys())) > 0 else np.inf
    if np.min(results) < current_min:
        chosen_predictors[run_count] = [current_predictors + [predictors[np.argmin(results)]], np.min(results)]
        predictors.pop(np.argmin(results))
        run_count += 1
    else: break
print("Chosen predictors:\n")
selected_variables = chosen_predictors[np.max(list(chosen_predictors.keys()))][0]
for predictor in selected_variables:
    print(predictor)
    print("\n")
print("Final model:")

if diagnostic_metric == "SECONDS scores":
    model = Logit(
            endog=y,
            exog=first_day_data[selected_variables]
            )
elif diagnostic_metric == "FOUR scores":
    model = OrderedModel(
            endog=y,
            exog=first_day_data[selected_variables],
            distr="logit"
            )
model_results = model.fit(method='bfgs')
print(model_results.summary())