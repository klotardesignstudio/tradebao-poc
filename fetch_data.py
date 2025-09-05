# -*- coding: utf-8 -*-

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_all_binance_data(symbol, interval, start_date_str):
    """
    Fetch all historical kline data for a Binance symbol, handling pagination backwards.

    :param symbol: The trading pair (e.g., "LINKBTC")
    :param interval: The timeframe (e.g., "1h", "4h", "1d")
    :param start_date_str: Start date string in format "YYYY-MM-DD"
    :return: A pandas DataFrame with historical OHLCV data.
    """
    base_url = "https://data-api.binance.vision"
    endpoint = "/api/v3/klines"
    limit = 1000 # Maximum candles per request

    # Convert the start date string to a millisecond timestamp
    start_timestamp = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)

    all_data = []

    print(f"Starting data fetch for {symbol} on timeframe {interval} from {start_date_str}...")

    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }

        # For the first request, do not set endTime. For subsequent requests, use the oldest candle time
        if all_data:
            # Use the time of the first candle from the last batch
            # as 'endTime' for the next request to avoid overlap
            params['endTime'] = all_data[0][0] - 1 # -1 to avoid fetching the same candle again

        try:
            response = requests.get(base_url + endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        if not data:
            # If the API returns no more data, stop the loop
            print("No more historical data to fetch.")
            break

        # Prepend the new batch to keep chronological order ascending
        all_data = data + all_data

        oldest_timestamp = data[0][0]
        oldest_date = datetime.fromtimestamp(oldest_timestamp / 1000)
        print(f"Batch of {len(data)} candles received. Oldest is: {oldest_date.strftime('%Y-%m-%d %H:%M:%S')}")

        if oldest_timestamp <= start_timestamp:
            # If we've reached or passed the start date, stop
            print("Start date reached. Fetch finished.")
            break

        # Small delay to avoid overloading Binance API
        time.sleep(0.5)

    if not all_data:
        return pd.DataFrame()

    # --- Final processing with Pandas ---
    columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
        "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume", "Ignore"
    ]
    df = pd.DataFrame(all_data, columns=columns)

    # Type cleaning and conversion
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')

    # Remove possible duplicates when stitching batches
    df.drop_duplicates(subset='Open Time', inplace=True)

    # Filter out rows before the exact start date
    df = df[df['Open Time'] >= pd.to_datetime(start_date_str)]

    df.set_index("Open Time", inplace=True)

    return df[["Open", "High", "Low", "Close", "Volume"]]

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

# Your new token list
tokens = [
    "1000CAT", "1000CHEEMS", "1000SATS", "1INCH", "1INCHDOWN", "1INCHUP", "1MBABYDOGE", "A", "A2Z", "AAVE", "AAVEDOWN", "AAVEUP", "ACA", "ACE", "ACH", "ACM", "ACT", "ACX", "ADA", "ADADOWN", "ADAUP", "ADX", "AE", "AERGO", "AEUR", "AEVO", "AGI", "AGIX", "AGLD", "AI", "AION", "AIXBT", "AKRO", "ALCX", "ALGO", "ALICE", "ALPACA", "ALPHA", "ALPINE", "ALT", "AMB", "AMP", "ANC", "ANIME", "ANKR", "ANT", "ANY", "APE", "API3", "APPC", "APT", "AR", "ARB", "ARDR", "ARK", "ARKM", "ARN", "ARPA", "ARS", "ASR", "AST", "ASTR", "ATA", "ATM", "ATOM", "AUCTION", "AUD", "AUDIO", "AUTO", "AVA", "AVAX", "AWE", "AXL", "AXS", "BABY", "BADGER", "BAKE", "BAL", "BANANA", "BANANAS31", "BAND", "BAR", "BAT", "BB", "BCC", "BCD", "BCH", "BCHA", "BCHABC", "BCHDOWN", "BCHSV", "BCHUP", "BCN", "BCPT", "BDOT", "BEAM", "BEAMX", "BEAR", "BEL", "BERA", "BETA", "BETH", "BFUSD", "BGBP", "BICO", "BIDR", "BIFI", "BIGTIME", "BIO", "BKRW", "BLUR", "BLZ", "BMT", "BNB", "BNBBEAR", "BNBBULL", "BNBDOWN", "BNBUP", "BNSOL", "BNT", "BNX", "BOME", "BOND", "BONK", "BOT", "BQX", "BRD", "BRL", "BROCCOLI714", "BSW", "BTC", "BTCB", "BTCDOWN", "BTCST", "BTCUP", "BTG", "BTS", "BTT", "BTTC", "BULL", "BURGER", "BUSD", "BVND", "BZRX", "C", "C98", "CAKE", "CATI", "CDT", "CELO", "CELR", "CETUS", "CFX", "CGPT", "CHAT", "CHESS", "CHR", "CHZ", "CITY", "CKB", "CLOAK", "CLV", "CMT", "CND", "COCOS", "COMBO", "COMP", "COOKIE", "COP", "COS", "COTI", "COVER", "COW", "CREAM", "CRV", "CTK", "CTSI", "CTXC", "CVC", "CVP", "CVX", "CYBER", "CZK", "D", "DAI", "DAR", "DASH", "DATA", "DCR", "DEGO", "DENT", "DEXE", "DF", "DGB", "DGD", "DIA", "DLT", "DNT", "DOCK", "DODO", "DOGE", "DOGS", "DOLO", "DOT", "DOTDOWN", "DOTUP", "DREP", "DUSK", "DYDX", "DYM", "EASY", "EDO", "EDU", "EGLD", "EIGEN", "ELF", "ENA", "ENG", "ENJ", "ENS", "EOS", "EOSBEAR", "EOSBULL", "EOSDOWN", "EOSUP", "EPIC", "EPS", "EPX", "ERA", "ERD", "ERN", "ETC", "ETH", "ETHBEAR", "ETHBULL", "ETHDOWN", "ETHFI", "ETHUP", "EUR", "EURI", "EVX", "EZ", "FARM", "FDUSD", "FET", "FIDA", "FIL", "FILDOWN", "FILUP", "FIO", "FIRO", "FIS", "FLM", "FLOKI", "FLOW", "FLUX", "FOR", "FORM", "FORTH", "FRONT", "FTM", "FTT", "FUEL", "FUN", "FXS", "G", "GAL", "GALA", "GAS", "GBP", "GFT", "GHST", "GLM", "GLMR", "GMT", "GMX", "GNO", "GNS", "GNT", "GO", "GPS", "GRS", "GRT", "GTC", "GTO", "GUN", "GVT", "GXS", "HAEDAL", "HARD", "HBAR", "HC", "HEGIC", "HEI", "HFT", "HIFI", "HIGH", "HIVE", "HMSTR", "HNT", "HOME", "HOOK", "HOT", "HSR", "HUMA", "HYPER", "ICN", "ICP", "ICX", "ID", "IDEX", "IDRT", "ILV", "IMX", "INIT", "INJ", "INS", "IO", "IOST", "IOTA", "IOTX", "IQ", "IRIS", "JASMY", "JOE", "JPY", "JST", "JTO", "JUP", "JUV", "KAIA", "KAITO", "KAVA", "KDA", "KEEP", "KERNEL", "KEY", "KLAY", "KMD", "KMNO", "KNC", "KP3R", "KSM", "LA", "LAYER", "LAZIO", "LDO", "LEND", "LEVER", "LINA", "LINK", "LINKDOWN", "LINKUP", "LISTA", "LIT", "LOKA", "LOOM", "LPT", "LQTY", "LRC", "LSK", "LTC", "LTCDOWN", "LTCUP", "LTO", "LUMIA", "LUN", "LUNA", "LUNC", "MAGIC", "MANA", "MANTA", "MASK", "MATIC", "MAV", "MBL", "MBOX", "MC", "MCO", "MDA", "MDT", "MDX", "ME", "MEME", "METIS", "MFT", "MINA", "MIR", "MITH", "MITO", "MKR", "MLN", "MOB", "MOD", "MOVE", "MOVR", "MTH", "MTL", "MUBARAK", "MULTI", "MXN", "NANO", "NAS", "NAV", "NBS", "NCASH", "NEAR", "NEBL", "NEIRO", "NEO", "NEWT", "NEXO", "NFP", "NGN", "NIL", "NKN", "NMR", "NOT", "NPXS", "NTRN", "NU", "NULS", "NXPC", "NXS", "OAX", "OCEAN", "OG", "OGN", "OLDSKY", "OM", "OMG", "OMNI", "ONDO", "ONE", "ONG", "ONT", "OOKI", "OP", "ORCA", "ORDI", "ORN", "OSMO", "OST", "OXT", "PARTI", "PAX", "PAXG", "PDA", "PENDLE", "PENGU", "PEOPLE", "PEPE", "PERL", "PERP", "PHA", "PHB", "PHX", "PIVX", "PIXEL", "PLA", "PLN", "PLUME", "PNT", "PNUT", "POA", "POE", "POL", "POLS", "POLY", "POLYX", "POND", "PORTAL", "PORTO", "POWR", "PPT", "PROM", "PROS", "PROVE", "PSG", "PUNDIX", "PYR", "PYTH", "QI", "QKC", "QLC", "QNT", "QSP", "QTUM", "QUICK", "RAD", "RAMP", "RARE", "RAY", "RCN", "RDN", "RDNT", "RED", "REEF", "REI", "REN", "RENBTC", "RENDER", "REP", "REQ", "RESOLV", "REZ", "RGT", "RIF", "RLC", "RNDR", "RON", "RONIN", "ROSE", "RPL", "RPX", "RSR", "RUB", "RUNE", "RVN", "S", "SAGA", "SAHARA", "SALT", "SAND", "SANTOS", "SC", "SCR", "SCRT", "SEI", "SFP", "SHELL", "SHIB", "SIGN", "SKL", "SLF", "SLP", "SNGLS", "SNM", "SNT", "SNX", "SOL", "SOLV", "SOMI", "SOPH", "SPARTA", "SPELL", "SPK", "SRM", "SSV", "STEEM", "STG", "STMX", "STO", "STORJ", "STORM", "STPT", "STRAT", "STRAX", "STRK", "STX", "SUB", "SUI", "SUN", "SUPER", "SUSD", "SUSHI", "SUSHIDOWN", "SUSHIUP", "SWRV", "SXP", "SXPDOWN", "SXPUP", "SXT", "SYN", "SYRUP", "SYS", "T", "TAO", "TCT", "TFUEL", "THE", "THETA", "TIA", "TKO", "TLM", "TNB", "TNSR", "TNT", "TOMO", "TON", "TORN", "TOWNS", "TRB", "TREE", "TRIBE", "TRIG", "TROY", "TRU", "TRUMP", "TRX", "TRXDOWN", "TRXUP", "TRY", "TST", "TURBO", "TUSD", "TUSDB", "TUT", "TVK", "TWT", "UAH", "UFT", "UMA", "UNFI", "UNI", "UNIDOWN", "UNIUP", "USD1", "USDC", "USDP", "USDS", "USDSB", "USDT", "UST", "USTC", "USUAL", "UTK", "VAI", "VANA", "VANRY", "VELODROME", "VEN", "VET", "VGX", "VIA", "VIB", "VIBE", "VIC", "VIDT", "VIRTUAL", "VITE", "VOXEL", "VTHO", "W", "WABI", "WAN", "WAVES", "WAXP", "WBETH", "WBTC", "WCT", "WIF", "WIN", "WING", "WINGS", "WLD", "WLFI", "WNXM", "WOO", "WPR", "WRX", "WTC", "XAI", "XEC", "XEM", "XLM", "XLMDOWN", "XLMUP", "XMR", "XNO", "XRP", "XRPBEAR", "XRPBULL", "XRPDOWN", "XRPUP", "XTZ", "XTZDOWN", "XTZUP", "XUSD", "XVG", "XVS", "XZC", "YFI", "YFIDOWN", "YFII", "YFIUP", "YGG", "YOYO", "ZAR", "ZEC", "ZEN", "ZIL", "ZK", "ZRO", "ZRX"
]

# Define the quote asset to use (usually USDT)
quote_asset = "USDT"

# Use list comprehension to build the full list of symbols
symbols = [f"{token.upper()}{quote_asset}" for token in tokens]

# Desired timeframe (1h = 1 hour, 4h = 4 hours, etc.)
interval = "4h"
# Start date for history
date_start = "2017-07-01"

# Test which symbols are available before fetching full data
symbols_available = symbols

# Local output directory (data/raw) relative to the project root
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'data', 'raw')
os.makedirs(output_dir, exist_ok=True)

if not symbols_available:
    print("\nNo available symbols to fetch data.")
else:
    print(f"\nStarting data fetch for {len(symbols_available)} available symbols...")
    for symbol in symbols_available:
        # Call the function to fetch the data
        df_symbol = fetch_all_binance_data(symbol, interval, date_start)

        if not df_symbol.empty:
            # Save the data to a new CSV file, indicating the timeframe
            name_file = os.path.join(output_dir, f'data_binance_{symbol}_{interval}.csv')
            df_symbol.to_csv(name_file)
            print(f"\n✅ SAVED! {len(df_symbol)} {interval} candles for {symbol.upper()} saved to '{name_file}'\n")
        else:
            print(f"\n❌ Failed to fetch data for {symbol.upper()}.\n")


import requests
import time

def test_symbols_availability(symbols, interval):
    """
    Test the availability of data for a list of symbols and timeframe.

    :param symbols: List of trading pairs (ex: ["LINKBTC", "ETHBTC"])
    :param interval: The timeframe (ex: "1h", "4h", "1d")
    :return: A list of symbols for which data could be fetched.
    """
    base_url = "https://data-api.binance.vision"
    endpoint = "/api/v3/klines"
    working_symbols = []

    print("Testing symbols availability...")

    for symbol in symbols:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": 1 # Only 1 candle for testing
        }
        try:
            response = requests.get(base_url + endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if data:
                print(f"✅ {symbol.upper()}: Data available.")
                working_symbols.append(symbol)
            else:
                print(f"❌ {symbol.upper()}: No data returned.")
        except requests.exceptions.RequestException as e:
            print(f"❌ {symbol.upper()}: Request error - {e}")

        time.sleep(0.1) # Small pause between tests

    print("Test of availability completed.")
    return working_symbols