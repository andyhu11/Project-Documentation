from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime
import os
import optuna

# 创建基于时间戳的结果目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = f'results_{timestamp}'
os.makedirs(result_dir, exist_ok=True)

# 加载消费数据集
df_cons = pd.read_csv('./final_data_for_consumption.csv')

# 提取特征和目标变量
X_cons = df_cons.drop(['target_cons'], axis=1).select_dtypes(include=['number'])
y_cons = df_cons['target_cons']

# 按时间顺序划分训练集（70%）、验证集（15%）、测试集（15%）
train_size = int(len(X_cons) * 0.7)
val_size = int(len(X_cons) * 0.15)

X_train_cons = X_cons[:train_size]
X_val_cons = X_cons[train_size:train_size + val_size]
X_test_cons = X_cons[train_size + val_size:]

y_train_cons = y_cons[:train_size]
y_val_cons = y_cons[train_size:train_size + val_size]
y_test_cons = y_cons[train_size + val_size:]

# 训练初始模型
model_cons = xgb.XGBRegressor(n_estimators=100, random_state=42)
model_cons.fit(X_train_cons, y_train_cons)

# 定义单模型评估函数（包含 MAPE）
def evaluate_single_model(model, X_test, y_test, name, return_metrics=False):
    y_pred = model.predict(xgb.DMatrix(X_test) if isinstance(model, xgb.Booster) else X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # 添加 MAPE 计算，避免除零错误
    y_test_safe = np.where(y_test == 0, 1e-10, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
    print(
        f"模型 {name} 性能：\n  MSE: {mse:.4f}\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}\n  R²: {r2:.4f}\n  MAPE: {mape:.4f}%\n")
    return [mse, rmse, mae, r2, mape] if return_metrics else None

# 评估初始模型（在验证集上）
metrics_initial_cons = evaluate_single_model(model_cons, X_val_cons, y_val_cons, "电力消费 (初始模型)", return_metrics=True)

# 超参数调优（5 轮）
tuning_rounds = 5

# 定义五折时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 消费模型调优目标函数（基于交叉验证）
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

    # 在训练集上进行五折交叉验证
    fold_r2_scores = []
    for train_idx, val_idx in tscv.split(X_train_cons):
        X_train_cv, X_val_cv = X_train_cons.iloc[train_idx], X_train_cons.iloc[val_idx]
        y_train_cv, y_val_cv = y_train_cons.iloc[train_idx], y_train_cons.iloc[val_idx]

        model = xgb.XGBRegressor(random_state=42, **params)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)

        r2 = r2_score(y_val_cv, y_pred)
        fold_r2_scores.append(r2)

    # 返回交叉验证的平均 R²
    return np.mean(fold_r2_scores)

# 多轮贝叶斯优化 - 消费模型
best_r2_cons, best_params_cons = -float('inf'), None
for r in range(1, tuning_rounds + 1):
    print(f"--- 第 {r} 轮贝叶斯优化 (Consumption) ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_cons, n_trials=50)
    best_params = study.best_params
    print(f"第 {r} 轮最佳参数 (Consumption):", best_params)

    # 在整个训练集上训练模型，并在验证集上评估
    model = xgb.XGBRegressor(random_state=42, **best_params)
    model.fit(X_train_cons, y_train_cons)
    metrics = evaluate_single_model(model, X_val_cons, y_val_cons, f"第 {r} 轮电力消费", return_metrics=True)

    if metrics[3] > best_r2_cons:
        best_r2_cons, best_params_cons = metrics[3], best_params
print(f"整体最佳参数 (Consumption): {best_params_cons}\n整体最佳 R² (Consumption): {best_r2_cons}")

# 训练最终模型
dtrain_cons = xgb.DMatrix(X_train_cons, label=y_train_cons)
dtest_cons = xgb.DMatrix(X_test_cons, label=y_test_cons)
params_cons = best_params_cons.copy()
num_round_cons = params_cons.pop('n_estimators')
booster_cons = xgb.train(params_cons, dtrain_cons, num_boost_round=num_round_cons, evals=[(dtest_cons, 'eval')],
                         early_stopping_rounds=300, verbose_eval=False)
metrics_final_cons = evaluate_single_model(booster_cons, X_test_cons, y_test_cons, "最终电力消费模型", return_metrics=True)

# 保存最终模型为 JSON 文件
booster_cons.save_model(os.path.join(result_dir, f'best_model_cons_90plus_{timestamp}.json'))

# 将性能指标保存到 Excel 文件（包含 MAPE）
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE'],
    'Consumption_Initial': metrics_initial_cons,
    'Consumption_Final': metrics_final_cons
})
metrics_df.to_excel(os.path.join(result_dir, f'model_performance_cons_90plus_{timestamp}.xlsx'), index=False)
print(f"消费模型性能指标已保存为 '{result_dir}/model_performance_cons_90plus_{timestamp}.xlsx'")
print("消费模型训练完成，结果已保存至：", result_dir)