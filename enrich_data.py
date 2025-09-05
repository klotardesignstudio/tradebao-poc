# -*- coding: utf-8 -*-
"""
# TRADEBAO POC - FEATURE ENGINEERING (Final Version)
Complete script to load raw data, compute technical indicators,
and create advanced features for the Machine Learning model.
"""

# 1. IMPORTS
import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# 2. TRADEBAO INDICATORS LIBRARY (Refined and Optimized)
# =============================================================================

def calculate_rsi(df, period=14):
    """Compute RSI using Exponential Moving Average (industry standard)."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(df, k=14, d=3):
    """Compute Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling(window=k).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=d).mean()
    return df

def calculate_cmf(df, period=20):
    """Compute Chaikin Money Flow."""
    # Avoid division by zero when High == Low
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    """Compute Bollinger Bands."""
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    std_dev = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + (std_dev * num_std_dev)
    df['BB_Lower'] = df['BB_Mid'] - (std_dev * num_std_dev)
    return df

def calculate_macd(df, fast_window=12, slow_window=26, signal_window=9):
    """Compute MACD."""
    ema_fast = df['Close'].ewm(span=fast_window, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_window, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def calculate_adx(df, window=14):
    """Compute ADX, DI+ and DI- (robust implementation)."""
    df_adx = df.copy()

    high_low = df_adx['High'] - df_adx['Low']
    high_close = np.abs(df_adx['High'] - df_adx['Close'].shift())
    low_close = np.abs(df_adx['Low'] - df_adx['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    move_up = df_adx['High'].diff()
    move_down = df_adx['Low'].diff()
    plus_dm = np.where((move_up > move_down) & (move_up > 0), move_up, 0)
    minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0)

    plus_dm_smooth = pd.Series(plus_dm, index=df_adx.index).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df_adx.index).ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    epsilon = 1e-10
    df['DI_Plus'] = 100 * (plus_dm_smooth / (atr + epsilon))
    df['DI_Minus'] = 100 * (minus_dm_smooth / (atr + epsilon))

    dx = 100 * (np.abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'] + epsilon))
    df['ADX'] = dx.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    if 'ATR' not in df.columns:
        df['ATR'] = atr

    return df

def create_advanced_features(df, percent_change_goal=0.03, look_forward_candles=5):
    """
    Create advanced, strategy-level features focused on mean-reversion behavior.
    """
    print("  -> Creating advanced strategic features...")

    # --- Context Features ---
    df['dist_SMA_200_pct'] = ((df['Close'] - df['SMA_200']) / df['SMA_200']) * 100

    df['macd_cross'] = 0
    bullish_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) < df['MACD_Signal'].shift(1))
    bearish_cross = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) > df['MACD_Signal'].shift(1))
    df.loc[bullish_cross, 'macd_cross'] = 1
    df.loc[bearish_cross, 'macd_cross'] = -1

    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour

    # --- New Strategic Features ---
    df['bb_width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['dist_sma_norm_by_atr'] = df['dist_SMA_200_pct'] / df['ATR']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['dist_sma_mom'] = df['dist_SMA_200_pct'].diff(periods=3)
    df['bb_pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['adx_trending'] = np.where(df['ADX'] > 25, 1, 0)

    # 4. Volume Spike: Is the current volume X times above recent average?
    # A value > 1.5, for example, indicates significantly high volume.
    df['volume_spike'] = df['Volume'] / df['Volume'].rolling(window=30).mean()

    # --- Target Variable ---
    future_highs = df['High'].shift(-look_forward_candles).rolling(window=look_forward_candles).max()
    df['target'] = np.where(future_highs >= df['Close'] * (1 + percent_change_goal), 1, 0)

    print("  -> Advanced features created.")
    return df

# =============================================================================
# 3. CONFIGURATION AND EXECUTION
# =============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(base_dir, 'data', 'raw')
enriched_data_dir = os.path.join(base_dir, 'data', 'enriched')
os.makedirs(enriched_data_dir, exist_ok=True)
timeframe_to_process = '4h'

tokens = [ "1000CAT", "1000CHEEMS", "1000SATS", "1INCH", "1INCHDOWN", "1INCHUP", "1MBABYDOGE", "A", "A2Z", "AAVE", "AAVEDOWN", "AAVEUP", "ACA", "ACE", "ACH", "ACM", "ACT", "ACX", "ADA", "ADADOWN", "ADAUP", "ADX", "AE", "AERGO", "AEUR", "AEVO", "AGI", "AGIX", "AGLD", "AI", "AION", "AIXBT", "AKRO", "ALCX", "ALGO", "ALICE", "ALPACA", "ALPHA", "ALPINE", "ALT", "AMB", "AMP", "ANC", "ANIME", "ANKR", "ANT", "ANY", "APE", "API3", "APPC", "APT", "AR", "ARB", "ARDR", "ARK", "ARKM", "ARN", "ARPA", "ARS", "ASR", "AST", "ASTR", "ATA", "ATM", "ATOM", "AUCTION", "AUD", "AUDIO", "AUTO", "AVA", "AVAX", "AWE", "AXL", "AXS", "BABY", "BADGER", "BAKE", "BAL", "BANANA", "BANANAS31", "BAND", "BAR", "BAT", "BB", "BCC", "BCD", "BCH", "BCHA", "BCHABC", "BCHDOWN", "BCHSV", "BCHUP", "BCN", "BCPT", "BDOT", "BEAM", "BEAMX", "BEAR", "BEL", "BERA", "BETA", "BETH", "BFUSD", "BGBP", "BICO", "BIDR", "BIFI", "BIGTIME", "BIO", "BKRW", "BLUR", "BLZ", "BMT", "BNB", "BNBBEAR", "BNBBULL", "BNBDOWN", "BNBUP", "BNSOL", "BNT", "BNX", "BOME", "BOND", "BONK", "BOT", "BQX", "BRD", "BRL", "BROCCOLI714", "BSW", "BTC", "BTCB", "BTCDOWN", "BTCST", "BTCUP", "BTG", "BTS", "BTT", "BTTC", "BULL", "BURGER", "BUSD", "BVND", "BZRX", "C", "C98", "CAKE", "CATI", "CDT", "CELO", "CELR", "CETUS", "CFX", "CGPT", "CHAT", "CHESS", "CHR", "CHZ", "CITY", "CKB", "CLOAK", "CLV", "CMT", "CND", "COCOS", "COMBO", "COMP", "COOKIE", "COP", "COS", "COTI", "COVER", "COW", "CREAM", "CRV", "CTK", "CTSI", "CTXC", "CVC", "CVP", "CVX", "CYBER", "CZK", "D", "DAI", "DAR", "DASH", "DATA", "DCR", "DEGO", "DENT", "DEXE", "DF", "DGB", "DGD", "DIA", "DLT", "DNT", "DOCK", "DODO", "DOGE", "DOGS", "DOLO", "DOT", "DOTDOWN", "DOTUP", "DREP", "DUSK", "DYDX", "DYM", "EASY", "EDO", "EDU", "EGLD", "EIGEN", "ELF", "ENA", "ENG", "ENJ", "ENS", "EOS", "EOSBEAR", "EOSBULL", "EOSDOWN", "EOSUP", "EPIC", "EPS", "EPX", "ERA", "ERD", "ERN", "ETC", "ETH", "ETHBEAR", "ETHBULL", "ETHDOWN", "ETHFI", "ETHUP", "EUR", "EURI", "EVX", "EZ", "FARM", "FDUSD", "FET", "FIDA", "FIL", "FILDOWN", "FILUP", "FIO", "FIRO", "FIS", "FLM", "FLOKI", "FLOW", "FLUX", "FOR", "FORM", "FORTH", "FRONT", "FTM", "FTT", "FUEL", "FUN", "FXS", "G", "GAL", "GALA", "GAS", "GBP", "GFT", "GHST", "GLM", "GLMR", "GMT", "GMX", "GNO", "GNS", "GNT", "GO", "GPS", "GRS", "GRT", "GTC", "GTO", "GUN", "GVT", "GXS", "HAEDAL", "HARD", "HBAR", "HC", "HEGIC", "HEI", "HFT", "HIFI", "HIGH", "HIVE", "HMSTR", "HNT", "HOME", "HOOK", "HOT", "HSR", "HUMA", "HYPER", "ICN", "ICP", "ICX", "ID", "IDEX", "IDRT", "ILV", "IMX", "INIT", "INJ", "INS", "IO", "IOST", "IOTA", "IOTX", "IQ", "IRIS", "JASMY", "JOE", "JPY", "JST", "JTO", "JUP", "JUV", "KAIA", "KAITO", "KAVA", "KDA", "KEEP", "KERNEL", "KEY", "KLAY", "KMD", "KMNO", "KNC", "KP3R", "KSM", "LA", "LAYER", "LAZIO", "LDO", "LEND", "LEVER", "LINA", "LINK", "LINKDOWN", "LINKUP", "LISTA", "LIT", "LOKA", "LOOM", "LPT", "LQTY", "LRC", "LSK", "LTC", "LTCDOWN", "LTCUP", "LTO", "LUMIA", "LUN", "LUNA", "LUNC", "MAGIC", "MANA", "MANTA", "MASK", "MATIC", "MAV", "MBL", "MBOX", "MC", "MCO", "MDA", "MDT", "MDX", "ME", "MEME", "METIS", "MFT", "MINA", "MIR", "MITH", "MITO", "MKR", "MLN", "MOB", "MOD", "MOVE", "MOVR", "MTH", "MTL", "MUBARAK", "MULTI", "MXN", "NANO", "NAS", "NAV", "NBS", "NCASH", "NEAR", "NEBL", "NEIRO", "NEO", "NEWT", "NEXO", "NFP", "NGN", "NIL", "NKN", "NMR", "NOT", "NPXS", "NTRN", "NU", "NULS", "NXPC", "NXS", "OAX", "OCEAN", "OG", "OGN", "OLDSKY", "OM", "OMG", "OMNI", "ONDO", "ONE", "ONG", "ONT", "OOKI", "OP", "ORCA", "ORDI", "ORN", "OSMO", "OST", "OXT", "PARTI", "PAX", "PAXG", "PDA", "PENDLE", "PENGU", "PEOPLE", "PEPE", "PERL", "PERP", "PHA", "PHB", "PHX", "PIVX", "PIXEL", "PLA", "PLN", "PLUME", "PNT", "PNUT", "POA", "POE", "POL", "POLS", "POLY", "POLYX", "POND", "PORTAL", "PORTO", "POWR", "PPT", "PROM", "PROS", "PROVE", "PSG", "PUNDIX", "PYR", "PYTH", "QI", "QKC", "QLC", "QNT", "QSP", "QTUM", "QUICK", "RAD", "RAMP", "RARE", "RAY", "RCN", "RDN", "RDNT", "RED", "REEF", "REI", "REN", "RENBTC", "RENDER", "REP", "REQ", "RESOLV", "REZ", "RGT", "RIF", "RLC", "RNDR", "RON", "RONIN", "ROSE", "RPL", "RPX", "RSR", "RUB", "RUNE", "RVN", "S", "SAGA", "SAHARA", "SALT", "SAND", "SANTOS", "SC", "SCR", "SCRT", "SEI", "SFP", "SHELL", "SHIB", "SIGN", "SKL", "SLF", "SLP", "SNGLS", "SNM", "SNT", "SNX", "SOL", "SOLV", "SOMI", "SOPH", "SPARTA", "SPELL", "SPK", "SRM", "SSV", "STEEM", "STG", "STMX", "STO", "STORJ", "STORM", "STPT", "STRAT", "STRAX", "STRK", "STX", "SUB", "SUI", "SUN", "SUPER", "SUSD", "SUSHI", "SUSHIDOWN", "SUSHIUP", "SWRV", "SXP", "SXPDOWN", "SXPUP", "SXT", "SYN", "SYRUP", "SYS", "T", "TAO", "TCT", "TFUEL", "THE", "THETA", "TIA", "TKO", "TLM", "TNB", "TNSR", "TNT", "TOMO", "TON", "TORN", "TOWNS", "TRB", "TREE", "TRIBE", "TRIG", "TROY", "TRU", "TRUMP", "TRX", "TRXDOWN", "TRXUP", "TRY", "TST", "TURBO", "TUSD", "TUSDB", "TUT", "TVK", "TWT", "UAH", "UFT", "UMA", "UNFI", "UNI", "UNIDOWN", "UNIUP", "USD1", "USDC", "USDP", "USDS", "USDSB", "USDT", "UST", "USTC", "USUAL", "UTK", "VAI", "VANA", "VANRY", "VELODROME", "VEN", "VET", "VGX", "VIA", "VIB", "VIBE", "VIC", "VIDT", "VIRTUAL", "VITE", "VOXEL", "VTHO", "W", "WABI", "WAN", "WAVES", "WAXP", "WBETH", "WBTC", "WCT", "WIF", "WIN", "WING", "WINGS", "WLD", "WLFI", "WNXM", "WOO", "WPR", "WRX", "WTC", "XAI", "XEC", "XEM", "XLM", "XLMDOWN", "XLMUP", "XMR", "XNO", "XRP", "XRPBEAR", "XRPBULL", "XRPDOWN", "XRPUP", "XTZ", "XTZDOWN", "XTZUP", "XUSD", "XVG", "XVS", "XZC", "YFI", "YFIDOWN", "YFII", "YFIUP", "YGG", "YOYO", "ZAR", "ZEC", "ZEN", "ZIL", "ZK", "ZRO", "ZRX" ]

# Define the quote asset to use (usually USDT)
quote_asset = "USDT"

# Use list comprehension to create the list of full symbols
symbols = [f"{token.upper()}{quote_asset}" for token in tokens]

# --- Summary counters ---
files_processed = 0
files_not_found = 0
files_with_few_rows = 0
# --------------------------------

for symbol in symbols:
    file_source = os.path.join(raw_data_dir, f'data_binance_{symbol}_{timeframe_to_process}.csv')
    file_destination = os.path.join(enriched_data_dir, f'{symbol}_{timeframe_to_process}.csv')

    if not os.path.exists(file_source):
        print(f"WARNING: Source file not found for {symbol}. Skipping.")
        files_not_found += 1
        continue

    try:
        df = pd.read_csv(file_source, index_col='Open Time', parse_dates=True)
        if len(df) < 250:
            print(f"WARNING: {symbol} has few rows ({len(df)}). Skipping to avoid errors.")
            files_with_few_rows += 1
            continue

        print(f"Processing {symbol}...")

        # ... (cálculos e criação de features) ...
        df['RSI_14'] = calculate_rsi(df, period=14)
        df = calculate_stochastic(df, k=14, d=3)
        df['CMF_20'] = calculate_cmf(df, period=20)
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df = calculate_bollinger_bands(df, window=20)
        df = calculate_macd(df)
        df = calculate_adx(df, window=14)
        df = create_advanced_features(df)

        df.to_csv(file_destination)
        print(f"-> {symbol} enriched and saved.")
        files_processed += 1

    except Exception as e:
        print(f"❗️Unexpected error processing {symbol}: {e}")

print("\n--- ENRICHMENT SUMMARY ---")
print(f"Total symbols in list: {len(symbols)}")
print(f"Successfully processed files: {files_processed}")
print(f"Missing source files in 'data/raw': {files_not_found}")
print(f"Files skipped due to < 250 rows: {files_with_few_rows}")
print("---------------------------------------------")