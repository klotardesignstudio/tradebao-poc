# -*- coding: utf-8 -*-
"""
# TRADEBAO - Trade Consultant v2.0 (Local)
Analyze a single token on demand and provide a trading recommendation based on
the optimized model. Local, Colab-free version.
"""

import pandas as pd
import numpy as np
import os
import joblib
import requests
from datetime import datetime
import json
import argparse

# =============================================================================
# 1. CONFIGURATION & PARAMETERS (Local)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_artifacts(artifacts_dir: str):
    """Load model, columns and threshold using fixed filenames."""
    model_path = os.path.join(artifacts_dir, "tradebao_model_xgb_trader.pkl")
    cols_path = os.path.join(artifacts_dir, "model_columns_xgb_trader.json")
    thr_path = os.path.join(artifacts_dir, "best_threshold.txt")

    if not (os.path.exists(model_path) and os.path.exists(cols_path) and os.path.exists(thr_path)):
        raise FileNotFoundError(
            "Artifacts not found. Expected files: 'tradebao_model_xgb_trader.pkl', "
            "'model_columns_xgb_trader.json', 'best_threshold.txt' in artifacts directory."
        )

    model = joblib.load(model_path)
    # XGBoost compatibility shims for cross-version pickles
    if not hasattr(model, 'use_label_encoder'):
        model.use_label_encoder = False
    if not hasattr(model, 'gpu_id'):
        model.gpu_id = 0
    if not hasattr(model, 'predictor'):
        model.predictor = 'auto'

    try:
        with open(cols_path, 'r') as f:
            columns = json.load(f)
    except Exception:
        columns = joblib.load(cols_path)

    with open(thr_path, 'r') as f:
        threshold = float(f.read())

    return model, columns, threshold


MODEL, MODEL_COLUMNS, THRESHOLD = load_artifacts(ARTIFACTS_DIR)

# Strategy parameters
STRATEGY_PARAMS = {
    "TP_PCT": 0.10,  # 10% Take Profit
    "SL_PCT": 0.15,  # 15% Stop-Loss
}

# =============================================================================
# 2. INDICATORS LIBRARY
# =============================================================================

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(df, k=14, d=3):
    low_min = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling(window=k).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=d).mean()
    return df


def calculate_cmf(df, period=20):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf


def calculate_bollinger_bands(df, window=20):
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    std_dev = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std_dev * 2)
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_adx(df, window=14):
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['DM+'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['DM-'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    TR_smooth = df['TR'].ewm(span=window, adjust=False).mean()
    DM_plus_smooth = df['DM+'].ewm(span=window, adjust=False).mean()
    DM_minus_smooth = df['DM-'].ewm(span=window, adjust=False).mean()
    df['DI_Plus'] = (DM_plus_smooth / TR_smooth) * 100
    df['DI_Minus'] = (DM_minus_smooth / TR_smooth) * 100
    df['DX'] = (abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])) * 100
    df['ADX'] = df['DX'].ewm(span=window, adjust=False).mean()
    return df


def create_advanced_features(df):
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df = calculate_adx(df)  # Dependency for ATR and ADX
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['dist_SMA_200_pct'] = (df['Close'] - df['SMA_200']) / df['SMA_200'] * 100
    df['macd_cross'] = np.where(df['MACD_Hist'] > 0, 1, 0)
    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour
    df['bb_width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['dist_sma_norm_by_atr'] = (df['Close'] - df['SMA_200']) / df['ATR']
    df['dist_sma_mom'] = df['dist_SMA_200_pct'].diff()
    df['bb_pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['adx_trending'] = np.where(df['ADX'] > 25, 1, 0)
    volume_mean = df['Volume'].rolling(window=30).mean()
    df['volume_spike'] = np.where(df['Volume'] > 2 * volume_mean, 1, 0)
    return df

# =============================================================================
# 3. TRADING LOGIC FUNCTIONS
# =============================================================================

def fetch_dados_recentes(symbol, interval='4h', limit=300):
    """Fetch recent kline data from Binance."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
            'Taker Buy Quote Asset Volume', 'Ignore'
        ])
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
        return df
    except requests.exceptions.RequestException:
        return None


def enriquecer_dados(df):
    """Add all required indicators to the DataFrame in the proper order."""
    if df is None or len(df) < 250:
        return None
    df_copy = df.copy()

    # Correct order to ensure dependencies are met
    df_copy['RSI_14'] = calculate_rsi(df_copy)
    df_copy = calculate_stochastic(df_copy)
    df_copy['CMF_20'] = calculate_cmf(df_copy)
    df_copy = calculate_bollinger_bands(df_copy)
    df_copy = calculate_macd(df_copy)
    df_copy = create_advanced_features(df_copy)  # Also calls calculate_adx

    df_copy.dropna(inplace=True)
    return df_copy


def analisar_token(symbol):
    print("-" * 60)
    print(f"Analyzing token: {symbol.upper()}")
    print("-" * 60)

    # 1. Fetch & enrich data
    print("Fetching recent data...")
    df_recent = fetch_dados_recentes(symbol)
    if df_recent is None:
        print(f"❌ Could not fetch data for {symbol.upper()}. Check if the symbol is valid.")
        return

    print("Calculating indicators...")
    df_enriched = enriquecer_dados(df_recent)
    if df_enriched is None:
        print("❌ Not enough data to calculate indicators.")
        return

    # 2. Get model opinion
    latest = df_enriched.iloc[[-1]]
    X_pred = latest.reindex(columns=MODEL_COLUMNS, fill_value=0)

    prob_buy = MODEL.predict_proba(X_pred)[:, 1][0]
    signal = "BUY" if prob_buy >= THRESHOLD else "NEUTRAL/SELL"
    current_price = latest['Close'].iloc[-1]

    # 3. Present the result
    print("\n--- TRADEBAO TECHNICAL OPINION ---")
    print(f"Model Confidence (Buy Prob.): {prob_buy:.2%}")
    print(f"Minimum Buy Threshold:        {THRESHOLD:.2%}")
    print(f"Recommended Signal:           {signal}")
    print("-" * 35)

    if signal == "BUY":
        print("\n--- SUGGESTED BUY ORDER ---")
        sl_price = current_price * (1 - STRATEGY_PARAMS['SL_PCT'])
        tp_price = current_price * (1 + STRATEGY_PARAMS['TP_PCT'])
        print(f"Entry Price (Market): {current_price:.8f}")
        print(f"Suggested Stop-Loss:  {sl_price:.8f} (-{STRATEGY_PARAMS['SL_PCT']:.0%})")
        print(f"Suggested Take-Profit:{tp_price:.8f} (+{STRATEGY_PARAMS['TP_PCT']:.0%})")
    else:
        print("\nNo buy order recommended at the moment.")

    # Bonus: show some indicators
    print("\n--- CURRENT INDICATORS ---")
    for col in ["RSI_14", "ADX", "bb_width", "%K"]:
        if col in latest:
            print(f"{col}: {latest[col].iloc[-1]:.2f}")
    if 'SMA_200' in latest:
        print(f"Price vs. 200SMA: {'ABOVE' if current_price > latest['SMA_200'].iloc[-1] else 'BELOW'}")


# =============================================================================
# 4. EXECUTION
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeBao Token Consultant (local)")
    parser.add_argument("--symbol", "-s", help="Token symbol, e.g., BTCUSDT")
    args = parser.parse_args()

    if args.symbol:
        analisar_token(args.symbol)
    else:
        while True:
            token_input = input("\nEnter the token symbol to analyze (e.g., BTCUSDT) or 'exit': ").strip()
            if token_input.lower() == 'exit':
                break
            if token_input:
                analisar_token(token_input)