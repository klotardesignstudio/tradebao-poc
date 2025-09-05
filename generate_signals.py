#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate buy signals for a list of tokens from data/tokens.json.
Loads artifacts from ./artifacts (tradebao_model_xgb_trader.pkl, model_columns_xgb_trader.json, best_threshold.txt),
fetches recent data from Binance, computes features, and prints the top-N tokens
with highest buy probability including Buy Prob and Recommended Signal.
"""

import os
import json
import argparse
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import requests
import joblib


# ===============================
# Paths & Constants
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
TOKENS_FILE = os.path.join(BASE_DIR, "data", "tokens.json")

DEFAULT_INTERVAL = "4h"
DEFAULT_LIMIT = 300


# ===============================
# Artifacts Loader
# ===============================
def load_artifacts(artifacts_dir: str) -> Tuple[object, List[str], float]:
    model_path = os.path.join(artifacts_dir, "tradebao_model_xgb_trader.pkl")
    cols_path = os.path.join(artifacts_dir, "model_columns_xgb_trader.json")
    thr_path = os.path.join(artifacts_dir, "best_threshold.txt")

    if not (os.path.exists(model_path) and os.path.exists(cols_path) and os.path.exists(thr_path)):
        raise FileNotFoundError(
            "Artifacts not found. Expected files: 'tradebao_model_xgb_trader.pkl', "
            "'model_columns_xgb_trader.json', 'best_threshold.txt' in artifacts directory."
        )

    model = joblib.load(model_path)
    # XGBoost compatibility shims
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


# ===============================
# Indicator Functions
# ===============================
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    low_min = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling(window=k).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=d).mean()
    return df


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    std_dev = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std_dev * 2)
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df = calculate_adx(df)
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


# ===============================
# Data Fetch & Enrichment
# ===============================
def fetch_recent_klines(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = DEFAULT_LIMIT) -> Optional[pd.DataFrame]:
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


def enrich(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 250:
        return None
    df = df.copy()
    df['RSI_14'] = calculate_rsi(df)
    df = calculate_stochastic(df)
    df['CMF_20'] = calculate_cmf(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = create_advanced_features(df)
    df.dropna(inplace=True)
    return df


# ===============================
# Main Logic
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Generate buy signals for tokens in data/tokens.json")
    parser.add_argument("--top", type=int, default=20, help="How many top tokens to display (default: 20)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval (default: 4h)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of klines to fetch (default: 300)")
    args = parser.parse_args()

    # Load artifacts
    model, model_columns, threshold = load_artifacts(ARTIFACTS_DIR)

    # Load tokens
    with open(TOKENS_FILE, 'r') as f:
        tokens = json.load(f)

    results = []
    for symbol in tokens:
        df = fetch_recent_klines(symbol, interval=args.interval, limit=args.limit)
        df_enriched = enrich(df)
        if df_enriched is None or df_enriched.empty:
            continue
        latest = df_enriched.iloc[[-1]]
        X_pred = latest.reindex(columns=model_columns, fill_value=0)
        try:
            prob_buy = model.predict_proba(X_pred)[:, 1][0]
        except Exception:
            continue
        signal = "BUY" if prob_buy >= threshold else "NEUTRAL/SELL"
        results.append({
            "symbol": symbol,
            "prob_buy": prob_buy,
            "signal": signal
        })

    if not results:
        print("No results to display.")
        return

    # Sort and display top N
    results.sort(key=lambda x: x["prob_buy"], reverse=True)
    top_n = results[: max(1, args.top)]

    print("\n=== Top Buy Signals ===")
    print(f"Showing top {len(top_n)} out of {len(results)} analyzed tokens (threshold: {threshold:.4f})")
    print("Rank  Symbol           Buy Prob   Recommend Signal")
    print("----  ---------------  --------   ----------------")
    for idx, item in enumerate(top_n, start=1):
        print(f"{idx:>4}  {item['symbol']:<15}  {item['prob_buy']*100:7.2f}%   {item['signal']}")


if __name__ == "__main__":
    main()


