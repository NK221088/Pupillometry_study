from NPI_investigation import left_NPI_data_cleaned, right_NPI_data_cleaned
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()
NPI_distribution_plots_path = os.getenv("NPI_distribution_plots_path")

random_state = 25

right_NPI_data_cleaned = right_NPI_data_cleaned.dropna()
right_NPI_data_cleaned.to_csv(
    os.path.join(NPI_distribution_plots_path, f'right_NPI_data_cleaned.csv'))

left_NPI_data_cleaned = left_NPI_data_cleaned.dropna()
left_NPI_data_cleaned.to_csv(
    os.path.join(NPI_distribution_plots_path, f'left_NPI_data_cleaned.csv'))

X = left_NPI_data_cleaned.copy()[[col for col in left_NPI_data_cleaned if col not in  ["NPi", "redcap repeat instance", "record id"]]]
y = left_NPI_data_cleaned.copy()["NPi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# Metrics
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train R²: {r2_train:.3f}")
print(f"Test  R²: {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test  RMSE: {rmse_test:.3f}")


coef_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print(coef_table)

residuals = y_test - y_test_pred

plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted NPi")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted NPi")
plt.savefig(
    os.path.join(NPI_distribution_plots_path, f'Residuals_vs_Predicted_NPi.pdf'),
    dpi=600,                     
    bbox_inches='tight',
    format='pdf'
)
plt.close()

X2 = X.copy()
X2["pupil_size_sq"] = X2["pupil size"] ** 2

X_tr, X_te, y_tr, y_te = train_test_split(
    X2, y, test_size=0.2, random_state=25
)

lin_reg = LinearRegression()
lin_reg.fit(X_tr, y_tr)

y_pred = lin_reg.predict(X_te)

# Predictions
y_train_pred = lin_reg.predict(X_tr)
y_test_pred = lin_reg.predict(X_te)

# Metrics
r2_train = r2_score(y_tr, y_train_pred)
r2_test = r2_score(y_te, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_tr, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_te, y_test_pred))

print(f"Train R²: {r2_train:.3f}")
print(f"Test  R²: {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test  RMSE: {rmse_test:.3f}")

coef_table = pd.DataFrame({
    "Feature": X_tr.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print(coef_table)

residuals = y_te - y_test_pred

plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted NPi")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted NPi with Pupil Size Squared")
plt.savefig(
    os.path.join(NPI_distribution_plots_path, f'Residuals_vs_Predicted_NPi_with_pupil_size_sq.pdf'),
    dpi=600,                     
    bbox_inches='tight',
    format='pdf'
)
plt.close()

X3 = X.copy()
X3["max_const_velocity_sq"] = X3["max const velocity"] ** 2

X_tr, X_te, y_tr, y_te = train_test_split(
    X3, y, test_size=0.2, random_state=25
)

lin_reg = LinearRegression()
lin_reg.fit(X_tr, y_tr)

y_pred = lin_reg.predict(X_te)

# Predictions
y_train_pred = lin_reg.predict(X_tr)
y_test_pred = lin_reg.predict(X_te)

# Metrics
r2_train = r2_score(y_tr, y_train_pred)
r2_test = r2_score(y_te, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_tr, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_te, y_test_pred))

print(f"Train R²: {r2_train:.3f}")
print(f"Test  R²: {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test  RMSE: {rmse_test:.3f}")

coef_table = pd.DataFrame({
    "Feature": X_tr.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print(coef_table)

residuals = y_te - y_test_pred

plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted NPi")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted NPi with Max Const Velocity Squared")
plt.savefig(
    os.path.join(NPI_distribution_plots_path, f'Residuals_vs_Predicted_NPi_with_max_const_velocity_sq.pdf'),
    dpi=600,                     
    bbox_inches='tight',
    format='pdf'
)
plt.close()

X4 = X.copy()
mu = X["pupil size"].median()
X4["pupil_size_centered_sq"] = (X4["pupil size"] - mu) ** 2

X_tr, X_te, y_tr, y_te = train_test_split(
    X4, y, test_size=0.2, random_state=25
)

lin_reg = LinearRegression()
lin_reg.fit(X_tr, y_tr)

y_pred = lin_reg.predict(X_te)

# Predictions
y_train_pred = lin_reg.predict(X_tr)
y_test_pred = lin_reg.predict(X_te)

# Metrics
r2_train = r2_score(y_tr, y_train_pred)
r2_test = r2_score(y_te, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_tr, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_te, y_test_pred))

print(f"Train R²: {r2_train:.3f}")
print(f"Test  R²: {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test  RMSE: {rmse_test:.3f}")

coef_table = pd.DataFrame({
    "Feature": X_tr.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print(coef_table)

residuals = y_te - y_test_pred

plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted NPi")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted NPi with centered Pupil Size Squared")
plt.savefig(
    os.path.join(NPI_distribution_plots_path, f'Residuals_vs_Predicted_NPi_with_centered_pupil_size_sq.pdf'),
    dpi=600,                     
    bbox_inches='tight',
    format='pdf'
)
plt.close()