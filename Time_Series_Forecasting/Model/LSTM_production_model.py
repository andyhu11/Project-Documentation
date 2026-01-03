import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. Set random seeds
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 2. Load data
data_path = 'C:/Users/29108/Desktop/能源优化/Final_data/final_data_for_production_scaled.csv'
data = pd.read_csv(data_path)

# 3. Prepare feature matrix X and target vector y
X = data.drop(columns=['target_prod']).values
y = data['target_prod'].values.reshape(-1, 1)


# 4. Define sliding window function
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# 5. Create time-step sequences
time_steps = 24
X_seq, y_seq = create_sequences(X, y, time_steps)

# 6. Split training, validation, and test sets in chronological order
train_size = int(len(X_seq) * 0.7)
val_size = int(len(X_seq) * 0.15)

X_train = X_seq[:train_size]
X_val = X_seq[train_size:train_size + val_size]
X_test = X_seq[train_size + val_size:]

y_train = y_seq[:train_size]
y_val = y_seq[train_size:train_size + val_size]
y_test = y_seq[train_size + val_size:]

# 7. Set the results output folder
save_dir = './results/production_baseline'
os.makedirs(save_dir, exist_ok=True)


# 8. Define a function to build the LSTM model (supports multi-layer, attention, custom units, and dropout)
def build_lstm_model(layers, units, dropout_rate, use_attention=False, learning_rate=0.001):
    inputs = Input(shape=(time_steps, X_train.shape[2]))
    x = inputs

    # Add multiple LSTM layers
    for i in range(layers - 1):
        x = LSTM(units, activation='tanh', return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)

    # The last LSTM layer: decide whether to return sequences based on whether attention is used
    if use_attention:
        x = LSTM(units, activation='tanh', return_sequences=True)(x)  # Return sequences for the attention layer
        x = Attention()([x, x])  # Self-attention mechanism, outputs 3D
        x = GlobalAveragePooling1D()(x)
    else:
        x = LSTM(units, activation='tanh', return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(1)(x)  # Output shape: (num_samples, 1)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


# 9. Define experiment configurations
experiments = [
    # One-layer LSTM comparison
    {'layers': 1, 'units': 64, 'dropout_rate': 0.2, 'use_attention': False, 'learning_rate': 0.001},
    # Two-layer LSTM comparison (different units, dropout rates, learning rates, attention)
    {'layers': 2, 'units': 32, 'dropout_rate': 0.2, 'use_attention': False, 'learning_rate': 0.001},
    {'layers': 2, 'units': 64, 'dropout_rate': 0.2, 'use_attention': False, 'learning_rate': 0.001},
    {'layers': 2, 'units': 128, 'dropout_rate': 0.2, 'use_attention': False, 'learning_rate': 0.001},
    {'layers': 2, 'units': 64, 'dropout_rate': 0.3, 'use_attention': False, 'learning_rate': 0.001},
    {'layers': 2, 'units': 64, 'dropout_rate': 0.5, 'use_attention': False, 'learning_rate': 0.001},
    {'layers': 2, 'units': 64, 'dropout_rate': 0.2, 'use_attention': False, 'learning_rate': 0.0001},
    {'layers': 2, 'units': 64, 'dropout_rate': 0.2, 'use_attention': True, 'learning_rate': 0.001},
    # Three-layer LSTM comparison
    {'layers': 3, 'units': 64, 'dropout_rate': 0.2, 'use_attention': False, 'learning_rate': 0.001},
]

# 10. 5-fold time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# 11. Train and evaluate each experiment configuration
best_config = None
best_avg_mse = float('inf')
cv_results = []

for idx, config in enumerate(experiments):
    print(f"\nEvaluating configuration {idx + 1}/{len(experiments)}: {config}")

    # Cross-validate for each configuration
    fold_mse_list = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

        # Build the model
        model_cv = build_lstm_model(**config)

        # Callbacks
        early_stop_cv = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')
        lr_schedule_cv = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        # Train the model
        model_cv.fit(X_train_cv, y_train_cv,
                     epochs=50,
                     batch_size=32,
                     validation_data=(X_val_cv, y_val_cv),
                     callbacks=[early_stop_cv, lr_schedule_cv],
                     verbose=0)

        # Validation set prediction
        y_pred_cv = model_cv.predict(X_val_cv, verbose=0)

        # Adjust the shape of predictions
        if y_pred_cv.ndim == 3:
            y_pred_cv = y_pred_cv[:, 0, :]  # Assume only the first time step prediction is needed

        # Compute metrics
        mse_cv = mean_squared_error(y_val_cv, y_pred_cv)
        fold_mse_list.append(mse_cv)
        print(f"  Fold {fold}: MSE={mse_cv:.6f}")

    # Compute average MSE
    avg_mse = np.mean(fold_mse_list)
    print(f"  Average MSE for configuration {idx + 1}: {avg_mse:.6f}")

    # Save cross-validation results
    cv_results.append([idx + 1, config, avg_mse])

    # Update best configuration
    if avg_mse < best_avg_mse:
        best_avg_mse = avg_mse
        best_config = config

# 12. Save cross-validation results
cv_df = pd.DataFrame(cv_results, columns=['Config_ID', 'Config', 'Avg_MSE'])
cv_df.to_csv(os.path.join(save_dir, 'cv_results_production_lstm.csv'), index=False)
print("✅ Cross-validation results saved!")

print(f"\nBest configuration: {best_config} with average MSE: {best_avg_mse:.6f}")

# 13. Train the final model using the best configuration
final_model = build_lstm_model(**best_config)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint_path = os.path.join(save_dir, 'best_lstm_model.keras')
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

# Train the final model
history = final_model.fit(X_train, y_train,
                          epochs=50,
                          batch_size=32,
                          validation_data=(X_val, y_val),
                          callbacks=[early_stop, lr_schedule, checkpoint],
                          verbose=1)

# 14. Test set evaluation
y_pred = final_model.predict(X_test, verbose=0)

# Adjust the shape of predictions (if needed)
if y_pred.ndim == 3:
    y_pred = y_pred[:, 0, :]  # Assume only the first time step prediction is needed

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
non_zero_idx = np.where(np.abs(y_test) > 1e-6)[0]
mape = np.mean(np.abs((y_test[non_zero_idx] - y_pred[non_zero_idx]) / y_test[non_zero_idx])) * 100
r2 = r2_score(y_test, y_pred)

print(f"\nTest set evaluation:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.6f}")

# 15. Save test set metrics
test_metrics = {
    'Metric': ['MSE', 'MAE', 'RMSE', 'MAPE', 'R2'],
    'Value': [mse, mae, rmse, mape, r2]
}
test_metrics_df = pd.DataFrame(test_metrics)
test_metrics_df.to_csv(os.path.join(save_dir, 'best_test_metrics_production_lstm.csv'), index=False)
print("✅ Best model test set metrics saved!")
