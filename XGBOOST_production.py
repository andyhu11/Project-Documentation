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

# 加载生产数据集
df_prod = pd.read_csv('./final_data_for_production.csv')

# 提取特征和目标变量
X_prod = df_prod.drop(['target_prod'], axis=1).select_dtypes(include=['number'])
y_prod = df_prod['target_prod']

# 按时间顺序划分训练集（70%）、验证集（15%）、测试集（15%）
train_size = int(len(X_prod) * 0.7)
val_size = int(len(X_prod) * 0.15)

X_train_prod = X_prod[:train_size]
X_val_prod = X_prod[train_size:train_size + val_size]
X_test_prod = X_prod[train_size + val_size:]

y_train_prod = y_prod[:train_size]
y_val_prod = y_prod[train_size:train_size + val_size]
y_test_prod = y_prod[train_size + val_size:]

# 训练初始模型
model_prod = xgb.XGBRegressor(n_estimators=100, random_state=42)
model_prod.fit(X_train_prod, y_train_prod)

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
metrics_initial_prod = evaluate_single_model(model_prod, X_val_prod, y_val_prod, "电力生产 (初始模型)", return_metrics=True)

# 超参数调优（5 轮）
tuning_rounds = 5

# 定义五折时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 生产模型调优目标函数（基于交叉验证）
def objective_prod(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 6, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),
        'subsample': trial.suggest_float('subsample', 0.9, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.85, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 0.001),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 9),
        'alpha': trial.suggest_float('alpha', 0.3, 0.5),
        'lambda': trial.suggest_float('lambda', 0.4, 0.6),
        'objective': 'reg:squarederror'
    }

    # 在训练集上进行五折交叉验证
    fold_r2_scores = []
    for train_idx, val_idx in tscv.split(X_train_prod):
        X_train_cv, X_val_cv = X_train_prod.iloc[train_idx], X_train_prod.iloc[val_idx]
        y_train_cv, y_val_cv = y_train_prod.iloc[train_idx], y_train_prod.iloc[val_idx]

        model = xgb.XGBRegressor(random_state=42, **params)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)

        r2 = r2_score(y_val_cv, y_pred)
        fold_r2_scores.append(r2)

    # 返回交叉验证的平均 R²
    return np.mean(fold_r2_scores)

# 多轮贝叶斯优化 - 生产模型
best_r2_prod, best_params_prod = -float('inf'), None
for r in range(1, tuning_rounds + 1):
    print(f"--- 第 {r} 轮贝叶斯优化 (Production) ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_prod, n_trials=50)
    best_params = study.best_params
    print(f"第 {r} 轮最佳参数 (Production):", best_params)

    # 在整个训练集上训练模型，并在验证集上评估
    model = xgb.XGBRegressor(random_state=42, **best_params)
    model.fit(X_train_prod, y_train_prod)
    metrics = evaluate_single_model(model, X_val_prod, y_val_prod, f"第 {r} 轮电力生产", return_metrics=True)

    if metrics[3] > best_r2_prod:
        best_r2_prod, best_params_prod = metrics[3], best_params
print(f"整体最佳参数 (Production): {best_params_prod}\n整体最佳 R² (Production): {best_r2_prod}")

# 训练最终模型
dtrain_prod = xgb.DMatrix(X_train_prod, label=y_train_prod)
dtest_prod = xgb.DMatrix(X_test_prod, label=y_test_prod)
params_prod = best_params_prod.copy()
num_round_prod = params_prod.pop('n_estimators')
booster_prod = xgb.train(params_prod, dtrain_prod, num_boost_round=num_round_prod, evals=[(dtest_prod, 'eval')],
                         early_stopping_rounds=300, verbose_eval=False)
metrics_final_prod = evaluate_single_model(booster_prod, X_test_prod, y_test_prod, "最终电力生产模型", return_metrics=True)

# 保存最终模型为 JSON 文件
booster_prod.save_model(os.path.join(result_dir, f'best_model_prod_90plus_{timestamp}.json'))

# 将性能指标保存到 Excel 文件（包含 MAPE）
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE'],
    'Production_Initial': metrics_initial_prod,
    'Production_Final': metrics_final_prod
})
metrics_df.to_excel(os.path.join(result_dir, f'model_performance_prod_90plus_{timestamp}.xlsx'), index=False)
print(f"生产模型性能指标已保存为 '{result_dir}/model_performance_prod_90plus_{timestamp}.xlsx'")
print("生产模型训练完成，结果已保存至：", result_dir)