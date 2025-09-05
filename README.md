# tradebao-poc
TradeBao is a Proof of Concept (PoC) for building, evaluating, and simulating crypto trading strategies end-to-end. It covers the full lifecycle:
- fetch historical OHLCV data from Binance
- enrich it with technical indicators and engineered features
- train an ML model and produce artifacts (model, columns, threshold)
- backtest strategies on enriched data
- generate real-time-like buy signals from live candles
- run a local paper trading loop using a simulated wallet

-> fetch_data
-> enrich_data
-> train_ml
-> backtest
-> token_consult
-> paper_trade

## Setup and Execution

### 1) Create a virtual environment

macOS/Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

### 3) Workflow (recommended order)

1. Fetch raw data (optional if you already have it)
```
python fetch_data.py
```
Outputs: `data/raw/dados_binance_{SYMBOL}_{INTERVAL}.csv`

2. Enrich data locally (create full feature set)
```
python enrich_data.py
```
Outputs: `data/enriched/analise_v2_{SYMBOL}_{INTERVAL}.csv`

3. Keep enriched data up-to-date (avoid gaps)
```
python update_enriched.py --interval 4h --limit 600
```
Merges the last candles and recomputes features over a rolling window.

4. Train the model (writes artifacts)
```
python train_ml.py
```
Artifacts (used by other scripts):
- `artifacts/tradebao_model.pkl` (or `tradebao_model_xgb_trader.pkl`)
- `artifacts/model_columns.json` (or `model_columns_xgb_trader.json`)
- `artifacts/best_threshold.txt`

5. Backtest (uses enriched data + artifacts)
```
python backtest.py
```
Outputs: `backtest_results/report_v*.html`, `backtest_results/trades_log_v*.csv`

6. Generate live-like signals (on-the-fly features)
```
python generate_signals.py --top 20
```
Shows the top N tokens by Buy Prob and Recommended Signal from `data/tokens.json`.

7. Token consult (single token)
```
python token_consult.py --symbol BTCUSDT
```

8. Paper trading (simulated wallet and execution)
```
python paper_trading.py
```
Reads tokens from `data/tokens.json`, writes wallet to `wallet/paper_trading_wallet.json`.