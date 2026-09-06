"""GRU direction model on 60x 1-min sequences -> P(up next 15m).

Per-step features (9): BTC 1m logret/std, |ret|, volume ratio (causal median),
range, ETH/SOL/XRP 1m logret/std, minute-of-hour sin/cos.
Return stds are fit on TRAIN-period 1m data only (no lookahead).
Saves: gru_state.pt, gru_meta.json (scaler+feats), gru_oof.pkl (time,p_gru), metrics.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from fetch_data import load_or_fetch
from signals import MODEL_DIR

SEQ = 60
FEATS_1M = ["btc_ret", "btc_abs", "btc_volr", "btc_rng",
            "eth_ret", "sol_ret", "xrp_ret", "min_sin", "min_cos"]
SEED = 7
torch.manual_seed(SEED)


def prep_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    med = df["volume"].rolling(1440, min_periods=120).median()
    df["volr"] = df["volume"] / med.replace(0, np.nan)
    df["rng"] = (df["high"] - df["low"]) / df["close"]
    return df


def build_sequences(days1m=120, days15m=150):
    btc15 = load_or_fetch("BTC", "15m", days=days15m)
    btc15["time"] = pd.to_datetime(btc15["time"], utc=True)
    m = {"BTC": prep_1m(load_or_fetch("BTC", "1m", days=days1m))}
    for a in ("ETH", "SOL", "XRP"):
        m[a] = prep_1m(load_or_fetch(a, "1m", days=days1m))
    for df in m.values():
        df["time"] = pd.to_datetime(df["time"], utc=True)

    grid = btc15[["time", "close"]].copy()
    grid["y"] = (grid["close"].shift(-1) > grid["close"]).astype(int)
    grid = grid.iloc[:-1].reset_index(drop=True)

    # train-period stds (first 70% of grid time span)
    cut = grid["time"].quantile(0.70)
    stds = {}
    for a in ("BTC", "ETH", "SOL", "XRP"):
        r = m[a].loc[m[a]["time"] < cut, "ret"].replace([np.inf, -np.inf], np.nan).dropna()
        stds[a] = float(r.std()) or 1e-4

    # numpy-backed blocks: last 60 closed 1m bars at each grid time
    B = m["BTC"]
    bt, bc, bv, bh, bl = (B["time"].values.astype("datetime64[ns]").astype(np.int64),
                          B["close"].values, B["volume"].values, B["high"].values, B["low"].values)
    bret = np.nan_to_num(np.r_[np.nan, np.diff(np.log(bc))])
    bvolr = bv / pd.Series(bv).rolling(1440, min_periods=120).median().replace(0, np.nan).values
    brng = (bh - bl) / bc
    aret = {}
    at = {}
    for a in ("ETH", "SOL", "XRP"):
        d = m[a]
        aret[a] = np.nan_to_num(np.r_[np.nan, np.diff(np.log(d["close"].values))])
        at[a] = d["time"].values.astype("datetime64[ns]").astype(np.int64)

    Gt = grid["time"].values.astype("datetime64[ns]").astype(np.int64)
    G60 = (Gt[:, None] - np.arange(SEQ - 1, -1, -1)[None, :] * 60_000_000_000)
    bi = np.searchsorted(bt, Gt, side="right")  # first 1m bar AFTER grid time
    # validity: 60th-back bar must be within the last 61 minutes
    valid = (bi >= SEQ) & (bt[bi - SEQ] > Gt - 61 * 60_000_000_000)
    idx = np.where(valid)[0]
    K = len(idx)
    bb = bi[idx]
    # gather BTC block fields: shape (K, SEQ)
    off = np.arange(SEQ)[None, :] + (bb - SEQ)[:, None]
    b_ret = bret[off] / stds["BTC"]
    b_volr = np.nan_to_num(bvolr[off], nan=1.0)
    b_rng = np.nan_to_num(brng[off], nan=0.0)
    ask = {}
    for a in ("ETH", "SOL", "XRP"):
        ai = np.searchsorted(at[a], G60[idx].ravel(), side="right") - 1
        ai = np.clip(ai, 0, len(aret[a]) - 1)
        ask[a] = (aret[a][ai] / stds[a]).reshape(K, SEQ)
    mins = pd.to_datetime(G60[idx].ravel()).minute.values.reshape(K, SEQ)
    X = np.stack([b_ret, np.abs(b_ret), b_volr, b_rng,
                  ask["ETH"], ask["SOL"], ask["XRP"],
                  np.sin(2 * np.pi * mins / 60), np.cos(2 * np.pi * mins / 60)], axis=-1)
    X = np.nan_to_num(X.astype(np.float32), nan=0.0)
    y = grid["y"].values[idx]
    ts = pd.to_datetime(pd.Series(Gt[idx]), utc=True).rename("time")
    return X, y, ts, stds


class GRU(nn.Module):
    def __init__(self, d_in=9, h=32, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(h, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(self.drop(h[-1])).squeeze(-1)


def main(epochs=25, batch=512, lr=1e-3):
    print("Building sequences...")
    X, y, ts, stds = build_sequences()
    n = len(X)
    a, b, c = int(n * 0.70), int(n * 0.775), int(n * 0.85)
    splits = {"train": (0, a), "calibA": (a, b), "calibB": (b, c), "test": (c, n)}
    print(f"n={n} " + str({k: v[1] - v[0] for k, v in splits.items()}))

    dev = "cpu"
    net = GRU().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X[:a]),
                                         torch.from_numpy(y[:a].astype(np.float32))),
                           batch_size=batch, shuffle=True)
    Xva, yva = torch.from_numpy(X[a:b]), y[a:b]

    best, bad, best_state = 1e9, 0, None
    for ep in range(epochs):
        net.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            lossf(net(xb.to(dev)), yb.to(dev)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva.to(dev))).numpy()
        vl = log_loss(yva, np.clip(pv, 1e-6, 1 - 1e-6))
        print(f"  epoch {ep + 1}: val logloss={vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pall = torch.sigmoid(net(torch.from_numpy(X).to(dev))).numpy()

    te = slice(c, n)
    m = {"n": n, "base_rate_test": round(float(y[te].mean()), 4),
         "test": {"acc": round(float(accuracy_score(y[te], pall[te] > 0.5)), 4),
                  "logloss": round(float(log_loss(y[te], np.clip(pall[te], 1e-6, 1 - 1e-6))), 4),
                  "brier": round(float(brier_score_loss(y[te], pall[te])), 4),
                  "auc": round(float(roc_auc_score(y[te], pall[te])), 4)},
         "test_start": str(ts.iloc[c])}
    print(json.dumps(m, indent=1))

    torch.save(best_state, MODEL_DIR / "gru_state.pt")
    (MODEL_DIR / "gru_meta.json").write_text(json.dumps(
        {"seq": SEQ, "feats": FEATS_1M, "stds": stds, "hidden": 32}, indent=1))
    pd.DataFrame({"time": ts, "p_gru": pall}).to_pickle(MODEL_DIR / "gru_oof.pkl")
    (MODEL_DIR / "gru_metrics.json").write_text(json.dumps(m, indent=1))
    print("saved.")
    return ts, pall


if __name__ == "__main__":
    main()
