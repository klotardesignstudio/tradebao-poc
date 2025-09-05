# -*- coding: utf-8 -*-
"""
# TRADEBAO POC - FINAL MODEL TRAINING (v2 - Fixed)
Train the final model using engineered features and the best hyperparameters
found by Optuna, with corrected feature selection logic.
"""

import pandas as pd
import numpy as np
import os, glob, joblib, json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =============================================================================
# 1. BEST PARAMETERS (FOUND BY OPTUNA)
# =============================================================================
OPTIMAL_FEATURES = [
    "Open", "Low", "Close", "RSI_14", "%K", "BB_Upper", "BB_Lower", "MACD",
    "MACD_Signal", "ADX", "ATR", "day_of_week", "bb_width", "dist_sma_norm_by_atr"
]

OPTIMAL_HYPERPARAMETERS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "n_jobs": -1, "random_state": 42,
    "learning_rate": 0.011782661304062865, "n_estimators": 650, "max_depth": 3,
    "subsample": 0.808341029867168, "colsample_bytree": 0.510424490430134,
    "min_child_weight": 6, "gamma": 1.1958488399707585,
}

OPTIMAL_THRESHOLD = 0.49

# =============================================================================
# 2. DATA LOADING
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
enriched_dir = os.path.join(base_dir, 'data', 'enriched')
artifacts_dir = os.path.join(base_dir, 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
timeframe_to_process = '4h'

print(f"Looking for files in: {enriched_dir}")
file_pattern = os.path.join(enriched_dir, f'*_{timeframe_to_process}.csv')
file_list = glob.glob(file_pattern)
if not file_list:
    raise FileNotFoundError(f"No files found matching pattern: {file_pattern}.")

print(f"Loading and concatenating {len(file_list)} files...")
# Load only necessary columns to reduce memory
required_columns = list(set(OPTIMAL_FEATURES + ['target']))
df = pd.concat(
    [pd.read_csv(f, usecols=lambda c: c in required_columns or c == 'Open Time') for f in file_list],
    ignore_index=True,
)
df['Open Time'] = pd.to_datetime(df['Open Time'])
df.set_index('Open Time', inplace=True)
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"DataFrame size after cleaning: {df.shape}")

# =============================================================================
# 3. DATA PREPARATION WITH OPTIMIZED FEATURES (FIXED)
# =============================================================================
y = df['target']

# Select only the winning features directly (keep original order)
print(f"\nSelecting the {len(OPTIMAL_FEATURES)} optimized features...")
X = df[OPTIMAL_FEATURES]

# Chronological split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Data split into {len(X_train)} train and {len(X_test)} test records.")

# =============================================================================
# 4. FINAL MODEL TRAINING
# =============================================================================
print("\nStarting final model training with best hyperparameters...")
model = xgb.XGBClassifier(**OPTIMAL_HYPERPARAMETERS)
model.fit(X_train, y_train)
print("Training completed!")

# =============================================================================
# 5. FINAL EVALUATION
# =============================================================================
print("\nMaking predictions on the test set...")
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

print("\n--- PERFORMANCE REPORT (Optimized Model) ---")
print(classification_report(y_test, y_pred, target_names=['No Buy (0)', 'Buy (1)']))
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy: {accuracy:.2%}")

# =============================================================================
# 6. SAVING FINAL ARTIFACTS
# =============================================================================
model_path = os.path.join(artifacts_dir, "tradebao_model.pkl")
columns_path = os.path.join(artifacts_dir, "model_columns.json")
threshold_path = os.path.join(artifacts_dir, "best_threshold.txt")

joblib.dump(model, model_path)
print(f"\n✅ Optimized model saved to: {model_path}")

with open(columns_path, "w") as f:
    json.dump(list(X.columns), f)
print("✅ Model columns saved.")

with open(threshold_path, "w") as f:
    f.write(str(OPTIMAL_THRESHOLD))
print(f"✅ Best threshold saved to: {threshold_path}")