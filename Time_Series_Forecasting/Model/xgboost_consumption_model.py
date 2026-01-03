from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime
import os
import optuna

# Create a timestamp-based results directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = f'results_{timestamp}'
os.makedirs(result_dir, exist_ok=True)

# Load the consumption dataset
df_cons = pd.read_csv('./final_data_for_consumption.csv')

# Extract features and target variable
X_cons = df_cons.drop(['target_cons'], axis=1).select_dtypes(include=['number'])
y_cons = df_cons['target_cons']

# Split into training (70%), validation (15%), and test (15%) sets in chronological order
train_size = int(len(X_cons) * 0.7)
val_size = int(len(X_cons) * 0.15)

X_train_cons = X_cons[:train_size]
X_val_cons = X_cons[train_size:train_size + val_size]
X_test_cons = X_cons[train_size + val_size:]

y_train_cons = y_cons[:train_size]
y_val_cons = y_cons[train_size:train_size + val_size]
y_test_cons = y_cons[train_size + val_size:]

# Train the initial model
model_cons = xgb.XGBRegressor(n_estimators=100, random_state=42)
model_cons.fit(X_train_cons, y_train_cons)

# Define a single-model evaluation function (including MAPE)
def evaluate_single_model(model, X_test, y_test, name, return_metrics=False):
    y_pred = model.predict(xgb.DMatrix(X_test) if isinstance(model, xgb.Booster) else X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Add MAPE calculation and avoid division-by-zero errors
    y_test_safe = np.where(y_test == 0, 1e-10, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
    print(
        f"Model {name} performance:\n  MSE: {mse:.4f}\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}\n  R²: {r2:.4f}\n  MAPE: {mape:.4f}%\n")
    return [mse, rmse, mae, r2, mape] if return_metrics else None

# Evaluate the initial model (on the validation set)
metrics_initial_cons = evaluate_single_model(model_cons, X_val_cons, y_val_cons, "Power Consumption (Initial Model)", return_metrics=True)

# Hyperparameter tuning (5 rounds)
tuning_rounds = 5

# Define 5-fold time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Consumption-model tuning objective function (based on cross-validation)
def objective_cons(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1700, 1800),
        'max_depth': trial.suggest_int('max_depth', 4, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.0125, 0.013),
        'subsample': trial.suggest_float('subsample', 0.97, 0.98),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.67, 0.68),
        'gamma': trial.suggest_float('gamma', 0.0003, 0.0004),
        'min_child_weight': trial.suggest_int('min_child_weight', 7, 8),
        'alpha': trial.suggest_float('alpha', 0.42, 0.45),
        'lambda': trial.suggest_float('lambda', 0.42, 0.44),
        'objective': 'reg:squarederror'
    }

    # Perform 5-fold cross-validation on the training set
    fold_r2_scores = []
    for train_idx, val_idx in tscv.split(X_train_cons):
        X_train_cv, X_val_cv = X_train_cons.iloc[train_idx], X_train_cons.iloc[val_idx]
        y_train_cv, y_val_cv = y_train_cons.iloc[train_idx], y_train_cons.iloc[val_idx]

        model = xgb.XGBRegressor(random_state=42, **params)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)

        r2 = r2_score(y_val_cv, y_pred)
        fold_r2_scores.append(r2)

    # Return the average cross-validated R²
    return np.mean(fold_r2_scores)

# Multi-round Bayesian optimization - consumption model
best_r2_cons, best_params_cons = -float('inf'), None
for r in range(1, tuning_rounds + 1):
    print(f"--- Bayesian Optimization Round {r} (Consumption) ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_cons, n_trials=50)
    best_params = study.best_params
    print(f"Best parameters in Round {r} (Consumption):", best_params)

    # Train the model on the full training set and evaluate on the validation set
    model = xgb.XGBRegressor(random_state=42, **best_params)
    model.fit(X_train_cons, y_train_cons)
    metrics = evaluate_single_model(model, X_val_cons, y_val_cons, f"Round {r} Power Consumption", return_metrics=True)

    if metrics[3] > best_r2_cons:
        best_r2_cons, best_params_cons = metrics[3], best_params
print(f"Overall best parameters (Consumption): {best_params_cons}\nOverall best R² (Consumption): {best_r2_cons}")

# Train the final model
dtrain_cons = xgb.DMatrix(X_train_cons, label=y_train_cons)
dtest_cons = xgb.DMatrix(X_test_cons, label=y_test_cons)
params_cons = best_params_cons.copy()
num_round_cons = params_cons.pop('n_estimators')
booster_cons = xgb.train(params_cons, dtrain_cons, num_boost_round=num_round_cons, evals=[(dtest_cons, 'eval')],
                         early_stopping_rounds=300, verbose_eval=False)
metrics_final_cons = evaluate_single_model(booster_cons, X_test_cons, y_test_cons, "Final Power Consumption Model", return_metrics=True)

# Save the final model as a JSON file
booster_cons.save_model(os.path.join(result_dir, f'best_model_cons_90plus_{timestamp}.json'))

# Save performance metrics to an Excel file (including MAPE)
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE'],
    'Consumption_Initial': metrics_initial_cons,
    'Consumption_Final': metrics_final_cons
})
metrics_df.to_excel(os.path.join(result_dir, f'model_performance_cons_90plus_{timestamp}.xlsx'), index=False)
print(f"Consumption model performance metrics have been saved to '{result_dir}/model_performance_cons_90plus_{timestamp}.xlsx'")
print("Consumption model training completed. Results saved to:", result_dir)
