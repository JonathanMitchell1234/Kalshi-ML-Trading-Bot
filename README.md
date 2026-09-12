# BitBot — BTC 15-min Gradient Boosting Predictor (for Kalshi)

Predicts Bitcoin's price 15 minutes ahead using LightGBM gradient boosting.
Data: free Binance.US 15m klines (no API key). No lookahead in features.

## Layout
- `src/fetch_data.py` — downloads/caches OHLC (`data_cache/btc_15m.csv`)
- `src/features.py` — 84 features: returns/momentum, SMA/EMA gaps, RSI, MACD,
  Bollinger, Stochastic, ATR/volatility, volume, UTC time encodings
- `src/train_gbm.py` — trains regression (next-15m log-return → price) +
  direction classifier, TimeSeriesSplit CV + 10% holdout
- `src/predict.py` — live inference + Kalshi strike edge helper
- `trained_models/` — `gbm_reg.pkl`, `gbm_clf.pkl`, `feature_cols.pkl`, `metrics.json`

## Usage
```bash
pip install -r requirements.txt
python src/train_gbm.py --days 365     # fetch + train (~2-3 min)
python src/predict.py                  # live next-15m prediction
python src/predict.py --strike 79700 --side yes --kalshi_yes_price 0.55
```

## Holdout results (last 10% ≈ 3,428 bars from 2026-07-30)
| metric | value |
|---|---|
| RMSE (price) | $146.78 vs $147.00 naive persistence — beats by 0.15% |
| MAE | $87.84 (~12 bps MAPE) |
| Directional accuracy (reg) | 50.4% |
| Classifier P(up) accuracy | 54.0% (base up-rate 47.2%) |
| R² (price) | 0.9996 — misleading; predicting ≈ last price always scores ~1 |

Honest read: 15-min BTC is close to a random walk. The model has a small
statistical edge on direction, not a crystal ball. For Kalshi: only bet when
`edge >= MIN_EDGE_THRESHOLD` (0.05), size via Kelly fraction (0.10), and log
live predictions vs outcomes to calibrate before using real money.

## Direction model v2 (calibrated P(up), per advisors' recommendation)
Reframed from point-price regression to binary direction, matching how Kalshi
binary brackets are priced. Only acts at |P−.5| ≥ 0.10 (the 60/40 rule).
```bash
python src/train_direction.py   # LightGBM + isotonic calibration (114 feats)
python src/train_gru.py         # GRU on 60x 1-min sequences (torch CPU)
python src/train_stack.py       # logistic stack + comparison
```
- Features (137): 15m technicals + 1-min microstructure + ETH/SOL/XRP leads +
  **Hyperliquid funding (level + z-score) + Coinbase premium + CVD/buyer-ratio/trade-velocity
  (taker fields rescued from the fetcher) + PCA market/idiosyncratic factors +
  Bybit↔HL funding spread (BTC only — helps BTC, hurts ETH, gated per asset)**.
  Multi-venue funding spreads investigated and closed: dYdX serves only the
  latest ~100 records, Kraken Futures has no funding-history endpoint, Bybit
  is geo-blocked (needs a user-provided proxy/VPN — then it's one env var).
  Thin-minute gaps → neutral-0 + known flags. 5-seed GBMs; isotonic-only on
  purged cB folds; return-weighted + deadband labels. Walk-forward validation
  with 3d purge + 5d embargo throughout.
- Held-out test: **BTC AUC 0.61, ETH AUC 0.55**; side-aware policy backtest on
  200 live Kalshi windows each: **BTC 86% (22 trades), ETH 60% (43)** —
  BTC's rate is a small-sample high (CI 65–97%); paper will regress it.
- **Walk-forward recalibration** (`train_direction --recalibrate`, daily cron):
  fresh isotonic on trailing-21d raw scores, ships `dir_cal_live.pkl` only on
  holdout-Brier improvement. Dashboard shows live vs base cal. Fixes the
  short-side overconfidence the paper log exposed (ETH 52% realized vs 65%
  implied on NO bets).
- SOL/XRP expansion tested and **gated out** (27%/38% backtests; thin markets,
  no taker fields on venue). BTC+ETH only in production.
- Held-out test: **BTC AUC 0.61, ETH AUC 0.60**; side-aware policy backtest on
  200 live Kalshi windows each: **BTC 56.5% (69 trades), ETH 60.9% (23)**.
- Lesson logged: an UP-framed hit metric once faked a crisis (down-heavy
  buckets read low); all conviction metrics are side-aware now.
## Live trading path (real money — gated, paper default)
- `src/live.py` — balance/positions/orders reads (safe anytime); `place_limit`
  / `cancel_order` / `cancel_all` hard-gated by `risk.require_live()`.
  2-key arming: `LIVE_TRADING=true` env **and** UI arm switch. No demo venue
  is reachable from here, so verification is read-only + preflight until funded.
- Preflight per order: 2-key armed, balance covers cost+fee, daily halt clear,
  exposure caps, no active halts. Every fill mirrored to paper DB
  (`exec_mode='live'`) + reconciliation vs exchange truth (`/api/live/status`).
- Dashboard Live section: venue, balance, positions, arm/disarm (typed
  confirm), cancel-all. Kill switch now also flattens resting orders.

## Sizing + evaluation + risk (institutional Phase 1)
- `src/sizing.py` — fractional Kelly on fee-net edge (tenth-Kelly default,
  2% bankroll cap, 10-contract cap). Both traders size every paper fill.
- `src/execution.py` — maker-mode paper (mid limits, maker fee, depth-capped
  fills; `EXEC_MODE`, default both with side-by-side P&L). Taker-fee float bug
  fixed (schedule-exact now). `src/selftest.py` runs at boot + `/api/health`.
- `src/evaluate.py` + `/api/eval` + dashboard section — per-model-version
  P&L, Wilson 95% CIs, equity/Sharpe/drawdown, live reliability.
- **MLflow tracking** (`src/mlflow_log.py`, store `sqlite:////Users/Jonathan/mlflow.db`,
  experiments `bitbot-crypto` / `bitbot-weather`): every train, recalibration,
  and backtest logs params/metrics/tags (feat-hash + top feats per run, so
  feature-set changes correlate with performance). Best-effort — training
  never fails on tracking errors.
- `src/risk.py` + `/api/risk` + `POST /api/halt-all` — daily-loss halt
  (−$500, auto-expires at UTC midnight), 10-position / $100-trade caps,
  calibration-drift halt (trailing-20 under 45% while implying ≥55%),
  stale-data + error-streak breakers, paper-mode lock (`require_live()`),
  one-button kill switch. `src/test_risk.py`: 20 checks, all passing.
- `python src/retrain_all.py` (weekly cron): refreshes wx + direction models
  and stamps `versions.json`; every paper trade carries its model version.
- Economics: edges and P&L are **net of the real Kalshi taker fee**
  (7¢×P×(1−P)/contract, July 2026 schedule) via `src/economics.py`.
- Platform note: torch and LightGBM segfault in one process on this machine,
  so GRU inference runs isolated in a child process (`signals.py`).

## Weather tab — Kalshi daily highs, NYC + Chicago (`/weather`)
`KXHIGHNY` / `KXHIGHCHI`: 6 active $1 brackets + tails per day, close 05:00 UTC,
settled on The Weather Company. Pipeline:
- Obs: **settlement-site stations** (NWS Central Park KNYC + O'Hare KORD via
  your agent string; IEM ASOS archive history through ~yesterday — far fresher
  than ERA5's 6-day lag) + ERA5 precip/cloud fill. GFS snapshots + 31-member
  ensemble logged every cycle for future blending.
- `src/train_wx.py` — GBM shootout per city (LightGBM vs XGBoost challenger,
  seeded): NYC LightGBM RMSE 6.1°F, CHI LightGBM 7.1°F (challenger flips by
  run — shootout decides each retrain). Tail-weighted training (extreme-
  anomaly upweight + post-hoc recalibration; `WX_TAIL_ALPHA`): small RMSE/
  Brier gains, brackets flat. Features now include humidity + wind-gust lags
  (small verified gain via ablation) alongside MSLP tendencies + soundings
  (NYC only).
- **Upper-air soundings** (`src/wx_raob.py`): observed KOKX/KILX profiles via
  IEM RAOB JSON (free, full history). NYC keeps them (RMSE 6.39→5.78,
  backtest 23%→25%, tendencies rank #3-5); CHI reverted (RMSE 7.02→7.33,
  KILX 200km offset — gate enforced per city). Serve: same-day 12Z obs or
  GFS-profile fallback (unit-checked, flagged); weather tab shows 850T.
- **Anomaly targets** (NYC only): predict T−clim, add back; RMSE 5.88→5.62,
  backtest 20%→25%, PIT uniform (KS p=0.17), CRPS 2.5. CHI tied → stays absolute.
- **2m dewpoint depression** (station history + live): NYC RMSE →5.79,
  CRPS →2.45, PIT KS 0.26. ECMWF IFS logged alongside GFS for future blending.
- **Diagnostics**: mean CRPS + PIT uniformity on every backtest; conformal
  intervals evaluated (kept BayesT); purge/embargo + blackout discipline on splits.
- **Calibration window 45d** (not 120d): fixes CHI PIT uniformity (KS p 0.008→0.13),
  CRPS 2.41→2.26 CHI / 2.45→2.39 NYC. Short windows track seasonal volatility.
- **Pooled multi-city challenger**: one GBM on NYC+CHI (city one-hot) wins
  test RMSE (5.82/6.94 vs 6.23/7.19) but NOT bracket backtests
  (18%/10% vs 20%/10%) — stays an experiment, per-city models serve.
- Randomized hyperparameter search: flat (±1%), params unchanged.
  `src/wx_calibrate.py` — **Bayesian Normal-Gamma** residual updater →
  predictive Student-t, **refit per prediction on trailing-90d errors**
  (σ adapts ~4.5 calm / ~8 volatile). Beat fixed-Gaussian on holdout Brier.
- `src/wx_backtest.py` — 60 settled events/city: NYC 28% argmax, CHI 12%.
  (CHI miss traced to frontal volatility + $1 bracket gaps, not bias:
  MAE 3.3°F.) `src/wx_trader.py` — 4×/day GFS cadence, best fee-aware YES
  edge ≥ 0.03 (0.05 for T+1 — forecast skill decays), 1/event, intraday
  conditioning on NWS observed max-so-far, plus an NWP-disagreement rail
  (skip when |model − NDFD| > 6°F, GFS fallback; NDFD logged every cycle).
- **STATUS 2026-09-09: WX PAPER HALTED (0/9 settled).** Post-mortem: every buy
  was a cold tail (below-X brackets) at +22–52% claimed edge; model runs 2–4°F
  colder than NWP-informed prices in heat regimes — a bias no threshold fixes.
  Re-entry: (1) GFS/NDFD blend on ≥30d logs (~Oct), backtest-gated; (2) fresh
  60-event bracket backtest ≥30% argmax. Logging + harness keep running.
- v1 limits (honest): no NWP input (GFS logged for v1.1 blending); ERA5-vs-TWC
  basis bias-corrected, residual reported. New modules only; crypto tab untouched.

## Live market: Kalshi 15-min BTC Up/Down (`KXBTC15M`)
One binary market per 15-min window (YES = expire ≥ target, target = price at
open, settled on CF BRTI). This matches the direction model exactly — no fair-value
mapping needed: fair YES = P(up), fair NO = 1 − P(up), edge net of taker fee.
- Auto-trader acts in the first 5 min of each window only (fresh signal),
  YES if P≥.6 / NO if P≤.4, edge ≥ 0.03 net of fees (≈2× the 1.75¢ max fee),
  1 position per event, BTC + ETH (`KXETH15M`, own LGBM: test AUC 0.58).
- Policy backtest on 200 settled windows per asset (current models):
  BTC traded 8 @ 37.5%, ETH traded 11 @ 45.5% — conviction edge is not
  reproducing in recent windows; live paper is the judge.
- Hourly $100 brackets (`KXBTC`) still shown for context; no longer traded.
- `src/backtest_15m.py`, `kalshi_client.get_15m/current_15m_ticker`,
  `auto_trader.pick_15m`, fee-aware NO side in `tracker`/`economics`.

## Dashboard (live Kalshi markets + paper-trade W/L)
```bash
python src/app.py        # -> http://127.0.0.1:8000
```
- Pulls **real** Kalshi `KXBTC` hourly bracket markets (188 $100-wide brackets
  per hour, live bid/ask) and overlays the model's fair value per bracket:
  Normal(pred, σ) mass in bracket, σ = holdout RMSE × √(min-to-expiry/15).
- **Paper Buy YES** records a paper trade at the live ask; open trades
  auto-settle against Kalshi's official `result` (win = +(100−paid)×n).
- **Backtest** replays settled events: signal 30 min before hourly expiry,
  predicted bracket vs Kalshi's settled winner (`result` + CF BRTI
  `expiration_value`). Latest: **12/48 brackets hit (25%)**, mean |pred−actual| $191.
- Modules: `src/kalshi_client.py` (real REST data + auth scaffolding),
  `src/signals.py` (fair values), `src/tracker.py` (SQLite W/L store),
  `src/backtest.py`, `src/app.py` + `src/static/dashboard.html`.
- Paper trading only — no real orders are placed.

## Auto paper-trader (24h hands-free run)
Each cycle (default every 20 min ≈ 72 decisions/day): settle open trades →
fresh model prediction → paper-buy the highest-edge YES bracket (ask 2–98¢,
edge ≥ 0.05) with guards (max 2 open/event, no double-buy, ≥5 min to close).
Skips are logged with reasons, so the audit trail is complete.
```bash
python src/auto_trader.py --once                 # one cycle now
python src/auto_trader.py --interval 20          # standalone loop
# or use Start/Stop on the dashboard (runs inside the web server)
```
`src/auto_trader.py` holds the engine; `auto_cycles` table in
`data_cache/paper_trades.db` holds every decision.
# Kalshi-ML-Trading-Bot
