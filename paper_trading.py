# -*- coding: utf-8 -*-
"""
# TRADEBAO LIVE TRADER v1.1 (Local)
# Objective: Run the optimized trading strategy in paper trading mode,
# reading the token list from a JSON file.
"""
import pandas as pd
import numpy as np
import os
import joblib
import json
import requests
from datetime import datetime
from time import sleep

# =============================================================================
# 1. GENERAL CONFIG & STRATEGY PARAMETERS (Local)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
TIMEFRAME = "4h"
MIN_TRADE_VALUE_USDT = 10.0

STRATEGY_PARAMS = {
    "TP_PCT": 0.10,
    "SL_PCT": 0.15,
    "TRAIL_PCT": 0.10,
    "ROCKET_ROC_WINDOW": 3,
    "ROCKET_ROC_THRESH": 0.06,
}

# Load token list from project data/tokens.json
TOKENS_FILE = os.path.join(BASE_DIR, "data", "tokens.json")
try:
    with open(TOKENS_FILE, 'r') as f:
        LISTA_DE_TOKENS = json.load(f)
    print(f"Loaded {len(LISTA_DE_TOKENS)} tokens from '{TOKENS_FILE}'.")
except FileNotFoundError:
    print(f"ERROR: Tokens file '{TOKENS_FILE}' not found.")
    LISTA_DE_TOKENS = []

# =============================================================================
# 2. FUNÇÕES DE INFRAESTRUTURA (CARREGAR/SALVAR/BUSCAR DADOS)
# =============================================================================
def load_or_initialize_wallet(filepath):
    """Load wallet or create a new one if not exists."""
    try:
        with open(filepath, 'r') as f:
            print(f"✅ Wallet found at '{filepath}'.")
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ Wallet file not found.")
        while True:
            try:
                valor_inicial = float(input("➡️ Enter initial cash to invest (USDT): "))
                if valor_inicial > 0:
                    new_wallet = {"cash_usdt": valor_inicial, "positions": []}
                    save_wallet(new_wallet, filepath)
                    return new_wallet
                else: print("Value must be positive.")
            except ValueError: print("Invalid input. Please enter a number.")

def save_wallet(wallet_data, filepath):
    """Save current wallet state to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(wallet_data, f, indent=4)
    print(f"💾 Wallet saved to '{filepath}'")

def fetch_dados_recentes(symbol, interval, limit=300):
    """Fetch recent kline data from Binance."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
        return df
    except requests.exceptions.RequestException:
        print(f"Error fetching data for {symbol}.")
        return None

# =============================================================================
# 3. FUNÇÕES DE LÓGICA DE TRADING
# =============================================================================

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


def enriquecer_dados_tempo_real(df: pd.DataFrame):
    """Aplica o mesmo pipeline de features usado por generate_signals/backtest."""
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

def gerar_sinal(model, threshold, df_enriquecido, colunas_modelo):
    """Generate signal (0 or 1) based on model and threshold."""
    if df_enriquecido is None or df_enriquecido.empty: return None, None
    dados_recentes = df_enriquecido.iloc[[-1]]
    dados_para_previsao = dados_recentes.reindex(columns=colunas_modelo, fill_value=0)
    prob_compra = model.predict_proba(dados_para_previsao)[:, 1][0]
    sinal = 1 if prob_compra >= threshold else 0
    return sinal, dados_recentes

def check_exit_conditions(position, df_recente, model, threshold, colunas_modelo):
    """Check all exit conditions for an open position."""
    preco_atual = df_recente['Close'].iloc[-1]
    preco_compra = position['preco_compra']
    pico_preco = position['peak_price']

    # Condition 1: Stop-Loss
    if preco_atual <= preco_compra * (1 - STRATEGY_PARAMS['SL_PCT']):
        return "STOP_LOSS"

    # Condition 2: Model signal turned off
    sinal, _ = gerar_sinal(model, threshold, df_recente, colunas_modelo)
    if sinal == 0:
        return "SIGNAL_OFF"

    # Condition 3: "Rocket Mode"
    if len(df_recente) > STRATEGY_PARAMS['ROCKET_ROC_WINDOW']:
        roc = (preco_atual / df_recente['Close'].iloc[-STRATEGY_PARAMS['ROCKET_ROC_WINDOW']] - 1)
        modo_foguete_ativo = roc >= STRATEGY_PARAMS['ROCKET_ROC_THRESH']
    else:
        modo_foguete_ativo = False

    if modo_foguete_ativo:
        # Condition 3a: Rocket trailing stop
        if preco_atual <= pico_preco * (1 - STRATEGY_PARAMS['TRAIL_PCT']):
            return "TRAILING_ROCKET"
    else:
        # Condition 3b: Standard take profit
        if preco_atual >= preco_compra * (1 + STRATEGY_PARAMS['TP_PCT']):
            return "TAKE_PROFIT"

    return None # Nenhuma condição de saída atendida

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
def main():
    print("Starting TradeBao Live Trader v1.1 (local)...")

    # Load model and parameters from local artifacts
    model_path = os.path.join(ARTIFACTS_DIR, "tradebao_model_xgb_trader.pkl")
    columns_path = os.path.join(ARTIFACTS_DIR, "model_columns_xgb_trader.json")
    threshold_path = os.path.join(ARTIFACTS_DIR, "best_threshold.txt")

    model = joblib.load(model_path)
    # XGBoost compatibility shims
    if not hasattr(model, 'use_label_encoder'):
        model.use_label_encoder = False
    if not hasattr(model, 'gpu_id'):
        model.gpu_id = 0
    if not hasattr(model, 'predictor'):
        model.predictor = 'auto'

    try:
        with open(columns_path, 'r') as f:
            colunas_modelo = json.load(f)
    except Exception:
        colunas_modelo = joblib.load(columns_path)

    with open(threshold_path, 'r') as f:
        threshold = float(f.read())

    wallet_path = os.path.join(BASE_DIR, 'wallet', 'paper_trading_wallet.json')
    wallet = load_or_initialize_wallet(wallet_path)
    houve_mudancas = False

    # --- PASSO 1: GERENCIAR E VENDER POSIÇÕES ABERTAS ---
    print("\n" + "="*60); print("STEP 1: Checking positions for SELL..."); print("="*60)
    posicoes_para_manter = []
    if not wallet.get("positions"):
        print("No open positions.")
    else:
        for pos in wallet["positions"]:
            token = pos["token"]
            print(f"--- Checking {token}...")
            df_recente = fetch_dados_recentes(token, TIMEFRAME)

            if df_recente is None:
                posicoes_para_manter.append(pos)
                continue

            # Atualiza o pico de preço da posição
            pos['peak_price'] = max(pos.get('peak_price', pos['preco_compra']), df_recente['High'].iloc[-1])

            df_enriquecido = enriquecer_dados_tempo_real(df_recente.copy())
            motivo_saida = check_exit_conditions(pos, df_enriquecido, model, threshold, colunas_modelo)

            if motivo_saida:
                houve_mudancas = True
                preco_venda = df_recente['Close'].iloc[-1]
                valor_venda = pos["quantidade"] * preco_venda
                lucro_prejuizo = ((preco_venda / pos["preco_compra"]) - 1) * 100
                wallet["cash_usdt"] += valor_venda
                print(f"🚨 SELL: {pos['quantidade']:.4f} of {token} at {preco_venda:.8f}.")
                print(f"   Reason: {motivo_saida} | Result: {lucro_prejuizo:.2f}%")
            else:
                print(f"✅ HOLD: {token}.")
                posicoes_para_manter.append(pos)

    wallet["positions"] = posicoes_para_manter

    # --- PASSO 2: PROCURAR E COMPRAR NOVAS POSIÇÕES ---
    print("\n" + "="*60); print("STEP 2: Searching for new BUYS..."); print("="*60)
    if wallet["cash_usdt"] < MIN_TRADE_VALUE_USDT:
        print(f"Insufficient cash ({wallet['cash_usdt']:.2f} USDT).")
    else:
        tokens_em_carteira = {p["token"] for p in wallet["positions"]}
        tokens_para_escanear = [t for t in LISTA_DE_TOKENS if t not in tokens_em_carteira]

        sinais_de_compra = []
        # NOVO: Variável para rastrear o melhor candidato
        melhor_candidato = {'token': None, 'proba': 0.0}

        for token in tokens_para_escanear:
            print(f"Analyzing {token.upper()}...", end='\r')
            df_recente = fetch_dados_recentes(token, TIMEFRAME)
            df_enriquecido = enriquecer_dados_tempo_real(df_recente.copy())

            # Vamos pegar a probabilidade diretamente
            if df_enriquecido is not None and not df_enriquecido.empty:
                dados_recentes = df_enriquecido.iloc[[-1]]
                dados_para_previsao = dados_recentes.reindex(columns=colunas_modelo, fill_value=0)
                prob_compra = model.predict_proba(dados_para_previsao)[:, 1][0]

                # Rastreia o token com a maior probabilidade
                if prob_compra > melhor_candidato['proba']:
                    melhor_candidato['token'] = token
                    melhor_candidato['proba'] = prob_compra

                # Verifica se atinge o nosso critério para comprar
                if prob_compra >= threshold:
                    sinais_de_compra.append({"token": token, "preco_compra": dados_recentes['Close'].iloc[0]})

        # NOVO: Imprime o resultado da análise
        print(f"\n[DEBUG] Best candidate: {melhor_candidato['token']} with {melhor_candidato['proba']:.4f} confidence.")
        print(f"        (Minimum buy threshold: {threshold:.4f})")

        if sinais_de_compra:
            houve_mudancas = True
            max_trades_possiveis = int(wallet["cash_usdt"] / MIN_TRADE_VALUE_USDT)
            # Aqui você poderia ordenar 'sinais_de_compra' por um score se quisesse
            sinais_para_executar = sinais_de_compra[:max_trades_possiveis]

            if len(sinais_para_executar) > 0:
                valor_por_trade = wallet["cash_usdt"] / len(sinais_para_executar)
                print(f"\nEXECUTING BUYS ({len(sinais_para_executar)} trades with {valor_por_trade:.2f} USDT each):")
                for sinal in sinais_para_executar:
                    token, preco_compra = sinal["token"], sinal["preco_compra"]
                    quantidade = valor_por_trade / preco_compra
                    wallet["positions"].append({
                        "token": token,
                        "preco_compra": preco_compra,
                        "quantidade": quantidade,
                        "peak_price": preco_compra # Inicia o pico com o preço de compra
                    })
                    wallet["cash_usdt"] -= valor_por_trade
                    print(f"  -> BUY: {quantidade:.4f} of {token.upper()} at {preco_compra:.8f}")

    # --- PASSO 3: SALVAR E EXIBIR ESTADO FINAL ---
    print("\n" + "="*60); print("STEP 3: Final Wallet State"); print("="*60)
    print(f"Available Cash: {wallet['cash_usdt']:.2f} USDT")
    print("\nOpen Positions:")
    if not wallet["positions"]:
        print("None.")
    else:
        for pos in wallet["positions"]:
            print(f"- {pos['token'].upper()}: {pos['quantidade']:.4f} unidades (Preço Médio: {pos['preco_compra']:.8f})")

    if houve_mudancas:
        save_wallet(wallet, wallet_path)
    else:
        print("\nNo changes to wallet in this run.")

if __name__ == "__main__":
    main()