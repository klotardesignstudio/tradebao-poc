# -*- coding: utf-8 -*-
"""
# TRADEBAO POC - ADVANCED BACKTESTING (v11 - Final Syntax)
Script with the final syntax fix for size_type, ensuring compatibility
with vectorbt's Numba engine.
"""
import pandas as pd
import joblib
import vectorbt as vbt
import os
import glob
import re

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
enriched_dir = os.path.join(base_dir, 'data', 'enriched')
backtest_dir = os.path.join(base_dir, 'backtest_results')
artifacts_dir = os.path.join(base_dir, 'artifacts')
os.makedirs(backtest_dir, exist_ok=True)

START_DATE_BACKTEST = '2024-07-01'
FINAL_DATE_BACKTEST   = '2024-12-31'
number_of_symbols_to_test = 690

current_version = 0
for filename in os.listdir(backtest_dir):
    if filename.startswith('trades_log_v'):
        result = re.findall(r'trades_log_v(\d+)', filename)
        if result:
            current_version = max(current_version, int(result[0]))
next_version = current_version + 1
print(f"This will be run v{next_version}.")

# =============================================================================
# 2. LOAD MODEL AND PREPARE DATA
# =============================================================================
print("Loading optimized model and threshold...")
model_path = os.path.join(artifacts_dir, 'tradebao_model_xgb_trader.pkl')
columns_path = os.path.join(artifacts_dir, 'model_columns_xgb_trader.json')
threshold_path = os.path.join(artifacts_dir, 'best_threshold.txt')

model = joblib.load(model_path)
import json
if not hasattr(model, 'use_label_encoder'):
    # Compatibility for models serialized with older/newer XGBoost versions
    model.use_label_encoder = False
if not hasattr(model, 'gpu_id'):
    # Avoid AttributeError in scikit-learn get_params when older params exist
    model.gpu_id = 0
if not hasattr(model, 'predictor'):
    # Ensure predictor param exists for get_params in current xgboost
    model.predictor = 'auto'
try:
    with open(columns_path, 'r') as f:
        model_columns = json.load(f)
except Exception:
    # Fallback: columns file might have been saved via joblib with .json name
    model_columns = joblib.load(columns_path)
with open(threshold_path, 'r') as f:
    threshold = float(f.read())
print(f"Using optimized threshold: {threshold:.4f}")

print(f"Loading {number_of_symbols_to_test} files for backtest...")
file_pattern = os.path.join(enriched_dir, '*_4h.csv')
file_list = glob.glob(file_pattern)[:number_of_symbols_to_test]

df = pd.concat([
    pd.read_csv(f, index_col='Open Time', parse_dates=True).assign(
        symbol=re.sub(r'^analise_v2_', '', os.path.basename(f)).rsplit('_', 1)[0]
    )
    for f in file_list
])
df.sort_index(inplace=True)
df.dropna(inplace=True)

def apply_maturation_period(group):
    start_date = group.index.min()
    cutoff_date = start_date + pd.Timedelta(days=90)
    return group[group.index >= cutoff_date]
df = df.groupby('symbol').apply(apply_maturation_period).reset_index(level=0, drop=True)
df.sort_index(inplace=True)

if START_DATE_BACKTEST:
    df = df[df.index >= pd.to_datetime(START_DATE_BACKTEST)]
if FINAL_DATE_BACKTEST:
    df = df[df.index <= pd.to_datetime(FINAL_DATE_BACKTEST)]

print(f"Data filtered. New DataFrame shape: {df.shape}")
X = df[model_columns]

# =============================================================================
# 3. GENERATE SIGNALS AND RUN BACKTEST
# =============================================================================
print("Generating probabilities and applying threshold to create signals...")
probabilities = model.predict_proba(X)[:, 1]
df['signal'] = (probabilities >= threshold).astype(int)

print("Starting backtest simulation with capital management...")
entries_raw = df.pivot_table(index=df.index, columns='symbol', values='signal')
exits_raw = df.pivot_table(index=df.index, columns='symbol', values='signal')
prices = df.pivot_table(index=df.index, columns='symbol', values='Close')

# Clean signals for Numba compatibility
entries = entries_raw.fillna(0).astype(bool)
exits = (exits_raw == 0).fillna(False).astype(bool)

portfolio = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
    init_cash=200,
    fees=0.001,
    size=1.0,
    size_type='Percent',
    sl_stop=0.15,
    freq='4H',
    group_by=True
)

# =============================================================================
# 4. SAVE VERSIONED RESULTS
# =============================================================================
print("\n--- AGGREGATED BACKTEST RESULTS (AGGRESSIVE EXPOSURE) ---")
print(portfolio.stats())

suffix_data = f"_{START_DATE_BACKTEST}_a_{FINAL_DATE_BACKTEST}" if START_DATE_BACKTEST and FINAL_DATE_BACKTEST else "_total"
graph_file_path = os.path.join(backtest_dir, f'report_v{next_version}{suffix_data}.html')
trades_file_path = os.path.join(backtest_dir, f'trades_log_v{next_version}{suffix_data}.csv')

portfolio.plot().write_html(graph_file_path)
print(f"\nPerformance chart saved to: {graph_file_path}")

trades_df = portfolio.trades.records_readable
trades_df.to_csv(trades_file_path)
print(f"Detailed log with {len(trades_df)} trades saved to: {trades_file_path}")