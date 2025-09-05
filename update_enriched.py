#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incrementally update data/enriched by fetching the most recent klines missing
from existing enriched files, recomputing features over a rolling window, and
merging into the CSV to avoid gaps.

Naming preserved: data/enriched/analise_v2_{SYMBOL}_{INTERVAL}.csv
Tokens source: data/tokens.json
"""

import os
import json
import argparse
from typing import Optional

import pandas as pd
import numpy as np
import requests


# ===============================
# Paths & Defaults
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICHED_DIR = os.path.join(BASE_DIR, 'data', 'enriched')
TOKENS_FILE = os.path.join(BASE_DIR, 'data', 'tokens.json')

DEFAULT_INTERVAL = '4h'
ROLLING_LIMIT = 600  # candles to fetch for proper indicator warmup


# ===============================
# Indicator Functions (same as generate_signals)
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
# Fetch & Enrich helpers
# ===============================
def fetch_recent_klines(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
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


def compute_enriched(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df['RSI_14'] = calculate_rsi(df)
    df = calculate_stochastic(df)
    df['CMF_20'] = calculate_cmf(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = create_advanced_features(df)
    df.dropna(inplace=True)
    return df


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description='Incrementally update enriched CSVs to avoid gaps')
    parser.add_argument('--interval', default=DEFAULT_INTERVAL, help='Interval to process (default: 4h)')
    parser.add_argument('--limit', type=int, default=ROLLING_LIMIT, help='Number of recent klines to fetch (default: 600)')
    parser.add_argument('--tokens', default=TOKENS_FILE, help='Path to tokens.json (default: data/tokens.json)')
    parser.add_argument('--dry-run', action='store_true', help='Do not write files, only show actions')
    args = parser.parse_args()

    os.makedirs(ENRICHED_DIR, exist_ok=True)

    with open(args.tokens, 'r') as f:
        tokens = json.load(f)

    updated = 0
    skipped = 0

    for symbol in tokens:
        fname = f"analise_v2_{symbol}_{args.interval}.csv"
        fpath = os.path.join(ENRICHED_DIR, fname)

        existing: Optional[pd.DataFrame] = None
        if os.path.exists(fpath):
            try:
                existing = pd.read_csv(fpath, index_col='Open Time', parse_dates=True)
            except Exception:
                existing = None

        # Fetch a rolling window and recompute features to ensure indicator correctness
        recent = fetch_recent_klines(symbol, args.interval, args.limit)
        if recent is None or recent.empty:
            skipped += 1
            continue

        new_enriched = compute_enriched(recent)
        if new_enriched is None or new_enriched.empty:
            skipped += 1
            continue

        if existing is not None and not existing.empty:
            # Merge by index, replace overlaps, keep chronological order
            merged = existing.combine_first(new_enriched)
            # Prefer the newest computations where overlapping
            merged.update(new_enriched)
            merged.sort_index(inplace=True)
        else:
            merged = new_enriched

        if args.dry_run:
            print(f"DRY-RUN: Would write {len(merged)} rows to {fpath}")
        else:
            merged.to_csv(fpath)
            print(f"Updated: {symbol} -> {fpath} ({len(merged)} rows)")
            updated += 1

    print(f"\nSummary: updated={updated}, skipped={skipped}, total_tokens={len(tokens)}")


if __name__ == '__main__':
    main()


