"""
train_model.py — GAGNN (Graph Attention Graph Neural Network)
Hong Kong 18-District AQHI Forecasting
=========================================
Architecture : GRU (temporal) → GAT×2 (spatial) → FC (output)
Input         : past 24 hours × 18 stations × N features
Output        : next 6 hours AQHI per station  [B, 18, 6]

CSV columns used
  AQHI_<district>   — 18 target columns
  HUM_<station>     — humidity sensors (mapped → district mean)
  PDIR_<station>    — wind direction 0-360°  (→ sin/cos encoded)
  WSPD_<station>    — wind speed km/h        (→ district mean)
  Cyclone_Present   — global binary flag

Run
  python train_model.py                  # full training
  python train_model.py --csv path/to/combined_2013_2026.csv
  python train_model.py --eval-only      # load best weight & print test MAE
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv

# ══════════════════════════════════════════════════════════════════════════════
# 1.  STATION / DISTRICT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# 18 AQHI monitoring districts with approximate centroid coordinates
DISTRICTS = [
    "Central/Western", "Eastern",      "Kwun Tong",    "Sham Shui Po",
    "Kwai Chung",      "Tsuen Wan",    "Yuen Long",    "Tuen Mun",
    "Tung Chung",      "Tai Po",       "Sha Tin",      "Tap Mun",
    "Causeway Bay",    "Central",      "Mong Kok",     "Tseung Kwan O",
    "Southern",        "North",
]

DISTRICT_COORDS = {
    "Central/Western": (22.2859, 114.1448),
    "Eastern":         (22.2820, 114.2210),
    "Kwun Tong":       (22.3130, 114.2240),
    "Sham Shui Po":    (22.3303, 114.1628),
    "Kwai Chung":      (22.3540, 114.1290),
    "Tsuen Wan":       (22.3710, 114.1140),
    "Yuen Long":       (22.4450, 114.0220),
    "Tuen Mun":        (22.3910, 113.9730),
    "Tung Chung":      (22.2890, 113.9430),
    "Tai Po":          (22.4510, 114.1650),
    "Sha Tin":         (22.3850, 114.1880),
    "Tap Mun":         (22.4710, 114.3610),
    "Causeway Bay":    (22.2800, 114.1840),
    "Central":         (22.2820, 114.1580),
    "Mong Kok":        (22.3200, 114.1690),
    "Tseung Kwan O":   (22.3070, 114.2590),
    "Southern":        (22.2470, 114.1580),
    "North":           (22.4960, 114.1380),
}

# Mountain barriers between district pairs — reduce edge weight by this factor
# (Pairs that straddle Lion Rock, Tai Mo Shan, or the central ridge)
BARRIER_PAIRS = {
    frozenset({"Sha Tin",    "Mong Kok"}):        0.30,
    frozenset({"Sha Tin",    "Kowloon City"}):     0.30,
    frozenset({"Kwai Chung", "Central/Western"}):  0.40,
    frozenset({"Tuen Mun",   "Tsuen Wan"}):        0.50,
    frozenset({"Tung Chung", "Tsuen Wan"}):        0.45,
    frozenset({"Tap Mun",    "Sha Tin"}):          0.40,
    frozenset({"North",      "Tai Po"}):           0.60,
}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def build_graph(districts=DISTRICTS, coords=DISTRICT_COORDS,
                barrier_pairs=BARRIER_PAIRS, max_dist_km=40.0):
    """
    Returns
      edge_index  : LongTensor [2, E]
      edge_weight : FloatTensor [E]   (inverse-distance-squared, barrier-adjusted)
    Fully connected graph; self-loops excluded; edges pruned beyond max_dist_km.
    """
    n = len(districts)
    src, dst, wts = [], [], []
    for i in range(n):
        lat1, lon1 = coords[districts[i]]
        for j in range(n):
            if i == j:
                continue
            lat2, lon2 = coords[districts[j]]
            d = _haversine_km(lat1, lon1, lat2, lon2)
            if d > max_dist_km:
                continue
            w = 1.0 / (d ** 2 + 1e-6)
            pair = frozenset({districts[i], districts[j]})
            w *= barrier_pairs.get(pair, 1.0)
            src.append(i)
            dst.append(j)
            wts.append(w)
    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts,        dtype=torch.float)
    return edge_index, edge_weight


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# Mapping: which weather station codes correspond to which district?
# Approximate mapping based on HKO station locations.
DISTRICT_HUM_MAP = {
    "Central/Western": ["HKO", "CCH"],
    "Eastern":         ["JKB", "SKG"],
    "Kwun Tong":       ["TKL", "JKB"],
    "Sham Shui Po":    ["HKO", "YCT"],
    "Kwai Chung":      ["KP",  "YCT"],
    "Tsuen Wan":       ["KSC", "TMS"],
    "Yuen Long":       ["SHA", "LFS"],
    "Tuen Mun":        ["TMS", "LFS"],
    "Tung Chung":      ["TC",  "HKA"],
    "Tai Po":          ["SSH", "SEK"],
    "Sha Tin":         ["SHA", "SEK"],
    "Tap Mun":         ["SKG", "SSH"],
    "Causeway Bay":    ["HKO", "JKB"],
    "Central":         ["HKO", "CCH"],
    "Mong Kok":        ["HKO", "YCT"],
    "Tseung Kwan O":   ["TKL", "SKG"],
    "Southern":        ["HKO", "PEN"],
    "North":           ["SSH", "SEK"],
}

DISTRICT_WIND_MAP = {
    "Central/Western": ["HKS", "SC"],
    "Eastern":         ["JKB", "SKG"],
    "Kwun Tong":       ["TKL", "JKB"],
    "Sham Shui Po":    ["CCH", "SC"],
    "Kwai Chung":      ["KP",  "NP"],
    "Tsuen Wan":       ["TPK", "NP"],
    "Yuen Long":       ["SHA", "LFS"],
    "Tuen Mun":        ["TME", "TUN"],
    "Tung Chung":      ["TC",  "HKA"],
    "Tai Po":          ["SSH", "SEK"],
    "Sha Tin":         ["SHA", "SHL"],
    "Tap Mun":         ["NGP", "WGL"],
    "Causeway Bay":    ["HKS", "SE"],
    "Central":         ["HKS", "PLC"],
    "Mong Kok":        ["CCB", "CCH"],
    "Tseung Kwan O":   ["TKL", "SKG"],
    "Southern":        ["WLP", "WGL"],
    "North":           ["SSH", "SEK"],
}


def _station_mean(df, prefix, stations, suffix=""):
    """Average available station columns with given prefix."""
    cols = [f"{prefix}{s}{suffix}" for s in stations
            if f"{prefix}{s}{suffix}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1)


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
      X : float32 array  [T, 18, num_features]
      y : float32 array  [T, 18]   (raw AQHI 1-11)

    Features per node (district):
      0  aqhi              current AQHI (normalised by scaler later)
      1  humidity_mean     mean of mapped HUM stations (%)
      2  wspd_mean         mean wind speed (km/h)
      3  wdir_sin          sin of mean wind direction
      4  wdir_cos          cos of mean wind direction
      5  cyclone           Cyclone_Present flag (0/1)
      6  hour_sin          sin(2π·hour/24)
      7  hour_cos          cos(2π·hour/24)
      8  dow_sin           sin(2π·dow/7)
      9  dow_cos           cos(2π·dow/7)
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── data cleaning ──────────────────────────────────────────────────────
    aqhi_cols = [f"AQHI_{d}" for d in DISTRICTS]
    df[aqhi_cols] = df[aqhi_cols].clip(1, 11)          # clip out-of-range
    hum_cols  = [c for c in df.columns if c.startswith("HUM_")]
    df[hum_cols] = df[hum_cols].clip(0, 100)           # humidity cap

    # fill NaN forward then backward
    df = df.ffill().bfill()

    # ── time features ───────────────────────────────────────────────────────
    hour = df["Date"].dt.hour.values
    dow  = df["Date"].dt.dayofweek.values
    h_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    h_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    d_sin = np.sin(2 * np.pi * dow  / 7 ).astype(np.float32)
    d_cos = np.cos(2 * np.pi * dow  / 7 ).astype(np.float32)
    cyclone = df["Cyclone_Present"].values.astype(np.float32) \
              if "Cyclone_Present" in df.columns \
              else np.zeros(len(df), dtype=np.float32)

    T  = len(df)
    N  = len(DISTRICTS)
    F  = 10
    X  = np.zeros((T, N, F), dtype=np.float32)
    y  = np.zeros((T, N),    dtype=np.float32)

    for ni, district in enumerate(DISTRICTS):
        aqhi = df[f"AQHI_{district}"].values.astype(np.float32)

        # humidity: average of mapped stations
        hum_stns = DISTRICT_HUM_MAP[district]
        hum = _station_mean(df, "HUM_", hum_stns).values.astype(np.float32)

        # wind speed: average of mapped stations
        w_stns = DISTRICT_WIND_MAP[district]
        wspd = _station_mean(df, "WSPD_", w_stns).values.astype(np.float32)

        # wind direction: average then sin/cos encode
        wdir_raw = _station_mean(df, "PDIR_", w_stns).values.astype(np.float32)
        wdir_rad = np.deg2rad(wdir_raw)
        wdir_sin = np.sin(wdir_rad).astype(np.float32)
        wdir_cos = np.cos(wdir_rad).astype(np.float32)

        X[:, ni, 0] = aqhi
        X[:, ni, 1] = hum
        X[:, ni, 2] = wspd
        X[:, ni, 3] = wdir_sin
        X[:, ni, 4] = wdir_cos
        X[:, ni, 5] = cyclone
        X[:, ni, 6] = h_sin
        X[:, ni, 7] = h_cos
        X[:, ni, 8] = d_sin
        X[:, ni, 9] = d_cos

        y[:, ni] = aqhi

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SLIDING WINDOW DATASET
# ══════════════════════════════════════════════════════════════════════════════

class AQHIDataset(torch.utils.data.Dataset):
    """
    Each sample:
      x_seq : [N, seq_len, F]   — history window
      y_seq : [N, horizon]      — future AQHI (normalised)
    """
    def __init__(self, X_norm, y_norm, seq_len=24, horizon=6):
        self.X = X_norm          # [T, N, F]
        self.y = y_norm          # [T, N]
        self.seq_len = seq_len
        self.horizon = horizon
        self.valid = len(X_norm) - seq_len - horizon + 1

    def __len__(self):
        return max(self.valid, 0)

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]               # [seq, N, F]
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0, 2)  # [N, seq, F]
        y_tgt = self.y[idx + self.seq_len : idx + self.seq_len + self.horizon]  # [horizon, N]
        y_tgt = torch.tensor(y_tgt, dtype=torch.float32).T          # [N, horizon]
        return x, y_tgt


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GAGNN(nn.Module):
    """
    Temporal  : 2-layer GRU per node
    Spatial   : 2× GAT with 4-head attention
    Output    : FC → [N, horizon]
    """
    def __init__(self, node_features=10, hidden_dim=128,
                 seq_len=24, horizon=6, gat_heads=4, dropout=0.2):
        super().__init__()
        self.seq_len  = seq_len
        self.horizon  = horizon
        self.hidden   = hidden_dim

        # ── temporal ────────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size  = node_features,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
        )

        # ── spatial (GAT) ───────────────────────────────────────────────────
        self.gat1 = GATConv(hidden_dim, hidden_dim,
                            heads=gat_heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim,
                            heads=gat_heads, concat=False, dropout=dropout)

        # ── output ──────────────────────────────────────────────────────────
        self.fc_out = nn.Linear(hidden_dim, horizon)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1     = nn.LayerNorm(hidden_dim)
        self.bn2     = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        """
        x          : [B, N, T, F]
        edge_index : [2, E]  (single-graph; will be batched internally)
        returns    : [B, N, horizon]
        """
        B, N, T, F_in = x.shape
        device = x.device

        # build batched edge index
        offsets = (torch.arange(B, device=device) * N).unsqueeze(1)  # [B,1]
        batched_ei = (edge_index.unsqueeze(0) + offsets.unsqueeze(2)) \
                     .view(2, -1)                                      # [2, B*E]

        # ── temporal phase ──────────────────────────────────────────────────
        x_flat = x.reshape(B * N, T, F_in)          # [B*N, T, F]
        gru_out, _ = self.gru(x_flat)               # [B*N, T, H]
        h = gru_out[:, -1, :]                        # [B*N, H]  last step

        # ── spatial phase ───────────────────────────────────────────────────
        h = self.bn1(self.relu(self.gat1(h, batched_ei)) + h)   # residual
        h = self.dropout(h)
        h = self.bn2(self.relu(self.gat2(h, batched_ei)) + h)   # residual

        # ── output ──────────────────────────────────────────────────────────
        out = self.fc_out(h)                         # [B*N, horizon]
        out = out.view(B, N, self.horizon)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 6.  NORMALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fit_scalers(X, y):
    """Fit per-feature scalers. X: [T,N,F], y: [T,N]."""
    T, N, F = X.shape

    # feature scaler: fit on [T*N, F]
    feat_scaler = StandardScaler()
    feat_scaler.fit(X.reshape(-1, F))

    # aqhi scaler: fit on [T*N, 1]
    aqhi_scaler = StandardScaler()
    aqhi_scaler.fit(y.reshape(-1, 1))

    return feat_scaler, aqhi_scaler


def apply_scalers(X, y, feat_scaler, aqhi_scaler):
    T, N, F = X.shape
    X_norm = feat_scaler.transform(X.reshape(-1, F)).reshape(T, N, F)
    y_norm = aqhi_scaler.transform(y.reshape(-1, 1)).reshape(T, N)
    return X_norm.astype(np.float32), y_norm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── load & combine CSVs ─────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if csv_path.is_dir():
        dfs = [pd.read_csv(p) for p in sorted(csv_path.glob("*.csv"))]
        df  = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Loaded {len(dfs)} CSV files from directory → {len(df)} rows")
    else:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    # ── feature engineering ─────────────────────────────────────────────────
    print("[INFO] Engineering features …")
    X, y = engineer_features(df)
    print(f"[INFO] Feature array shape: {X.shape}  Target shape: {y.shape}")

    # ── chronological split (80/10/10) ──────────────────────────────────────
    T  = len(X)
    t1 = int(T * 0.80)
    t2 = int(T * 0.90)

    X_tr, y_tr = X[:t1],    y[:t1]
    X_va, y_va = X[t1:t2],  y[t1:t2]
    X_te, y_te = X[t2:],    y[t2:]

    # ── scalers ─────────────────────────────────────────────────────────────
    feat_scaler, aqhi_scaler = fit_scalers(X_tr, y_tr)
    joblib.dump(feat_scaler,  args.out_dir / "feat_scaler.pkl")
    joblib.dump(aqhi_scaler,  args.out_dir / "aqhi_scaler.pkl")
    print("[INFO] Scalers saved.")

    X_tr_n, y_tr_n = apply_scalers(X_tr, y_tr, feat_scaler, aqhi_scaler)
    X_va_n, y_va_n = apply_scalers(X_va, y_va, feat_scaler, aqhi_scaler)
    X_te_n, y_te_n = apply_scalers(X_te, y_te, feat_scaler, aqhi_scaler)

    # ── high-risk oversampling (AQHI ≥ 8) ───────────────────────────────────
    high_mask = (y_tr >= 8).any(axis=1)       # rows where any station ≥ 8
    hi_count  = high_mask.sum()
    if hi_count > 0:
        repeats = max(1, int(len(X_tr_n) * 0.15 / hi_count))
        X_hi = np.repeat(X_tr_n[high_mask], repeats, axis=0)
        y_hi = np.repeat(y_tr_n[high_mask], repeats, axis=0)
        X_tr_n = np.concatenate([X_tr_n, X_hi], axis=0)
        y_tr_n = np.concatenate([y_tr_n, y_hi], axis=0)
        print(f"[INFO] Oversampled {hi_count} high-risk rows ×{repeats} "
              f"→ training set now {len(X_tr_n)} rows")

    # ── datasets & loaders ──────────────────────────────────────────────────
    SEQ_LEN = args.seq_len
    HORIZON = args.horizon

    ds_tr = AQHIDataset(X_tr_n, y_tr_n, SEQ_LEN, HORIZON)
    ds_va = AQHIDataset(X_va_n, y_va_n, SEQ_LEN, HORIZON)
    ds_te = AQHIDataset(X_te_n, y_te_n, SEQ_LEN, HORIZON)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = torch.utils.data.DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── graph ────────────────────────────────────────────────────────────────
    edge_index, edge_weight = build_graph()
    edge_index = edge_index.to(device)
    print(f"[INFO] Graph: {len(DISTRICTS)} nodes, {edge_index.shape[1]} directed edges")

    # ── model, optimiser, scheduler ─────────────────────────────────────────
    model = GAGNN(
        node_features = X.shape[2],
        hidden_dim    = args.hidden_dim,
        seq_len       = SEQ_LEN,
        horizon       = HORIZON,
        gat_heads     = 4,
        dropout       = 0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {total_params:,}")

    criterion  = nn.L1Loss()                               # MAE — physical meaning
    optimiser  = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # ── training loop ────────────────────────────────────────────────────────
    best_val_mae  = float("inf")
    early_stop_ct = 0
    best_weight   = args.out_dir / "hk_pro_model_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        tr_loss = 0.0
        for x_batch, y_batch in dl_tr:
            x_batch = x_batch.to(device)   # [B, N, T, F]
            y_batch = y_batch.to(device)   # [B, N, horizon]
            optimiser.zero_grad()
            pred = model(x_batch, edge_index)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item()
        tr_loss /= len(dl_tr)

        # ── validate ──
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in dl_va:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred  = model(x_batch, edge_index)
                va_loss += criterion(pred, y_batch).item()
        va_loss /= len(dl_va)

        scheduler.step(va_loss)

        # convert back to AQHI scale for interpretable MAE
        scale = aqhi_scaler.scale_[0]
        tr_mae_aqhi = tr_loss * scale
        va_mae_aqhi = va_loss * scale

        elapsed = time.time() - t0
        print(f"Epoch {epoch:04d}/{args.epochs}  "
              f"Train MAE={tr_mae_aqhi:.4f}  Val MAE={va_mae_aqhi:.4f}  "
              f"lr={optimiser.param_groups[0]['lr']:.2e}  {elapsed:.1f}s")

        # ── checkpoint ──
        if va_loss < best_val_mae:
            best_val_mae = va_loss
            early_stop_ct = 0
            torch.save(model.state_dict(), best_weight)
        else:
            early_stop_ct += 1
            if early_stop_ct >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── test evaluation ──────────────────────────────────────────────────────
    print("\n[INFO] Loading best weights for test evaluation …")
    model.load_state_dict(torch.load(best_weight, map_location=device))
    model.eval()

    te_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in dl_te:
            x_batch = x_batch.to(device)
            pred = model(x_batch, edge_index)
            te_loss += criterion(pred, y_batch.to(device)).item()
            # denormalise for per-district reporting
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    te_loss /= len(dl_te)
    te_mae_aqhi = te_loss * scale

    all_preds   = np.concatenate(all_preds,   axis=0)  # [samples, N, horizon]
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"\n{'='*60}")
    print(f"  Test MAE (normalised scale) : {te_loss:.6f}")
    print(f"  Test MAE (AQHI scale)       : {te_mae_aqhi:.4f}")
    print(f"{'='*60}")

    # per-district breakdown
    print("\nPer-District Test MAE (AQHI units):")
    for ni, d in enumerate(DISTRICTS):
        preds_d   = all_preds[:, ni, :] * scale + aqhi_scaler.mean_[0]
        targets_d = all_targets[:, ni, :] * scale + aqhi_scaler.mean_[0]
        mae_d = np.abs(preds_d - targets_d).mean()
        print(f"  {d:<20s} {mae_d:.3f}")

    print(f"\n[INFO] Best model saved → {best_weight}")
    return model, feat_scaler, aqhi_scaler


# ══════════════════════════════════════════════════════════════════════════════
# 8.  INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_model_for_inference(weight_path, feat_scaler_path, aqhi_scaler_path,
                              node_features=10, hidden_dim=128,
                              seq_len=24, horizon=6, device=None):
    """Load saved model + scalers for Render / production inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_scaler  = joblib.load(feat_scaler_path)
    aqhi_scaler  = joblib.load(aqhi_scaler_path)

    model = GAGNN(node_features=node_features, hidden_dim=hidden_dim,
                  seq_len=seq_len, horizon=horizon).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    edge_index, _ = build_graph()
    edge_index = edge_index.to(device)

    return model, feat_scaler, aqhi_scaler, edge_index


def predict(model, feat_scaler, aqhi_scaler, edge_index,
            recent_df: pd.DataFrame, seq_len=24, horizon=6,
            device=None) -> pd.DataFrame:
    """
    recent_df : last `seq_len` hours of raw CSV data (≥ seq_len rows)
    Returns   : DataFrame with columns = DISTRICTS, index = forecast hours 1…horizon
    """
    if device is None:
        device = next(model.parameters()).device

    X, _ = engineer_features(recent_df.tail(seq_len + horizon))
    X    = X[-seq_len:]                                       # [seq, N, F]
    T, N, F = X.shape
    X_norm = feat_scaler.transform(X.reshape(-1, F)).reshape(T, N, F)

    x_tensor = torch.tensor(X_norm, dtype=torch.float32) \
                     .unsqueeze(0)                            # [1, N, seq, F]
    x_tensor = x_tensor.permute(0, 2, 1, 3)                  # wait — dataset gives [N,T,F]
    # fix: dataset already returns [N, T, F], model expects [B, N, T, F]
    x_tensor = torch.tensor(X_norm, dtype=torch.float32) \
                     .permute(1, 0, 2).unsqueeze(0)           # [1, N, T, F]

    with torch.no_grad():
        pred_norm = model(x_tensor.to(device), edge_index)   # [1, N, horizon]
    pred_norm = pred_norm.squeeze(0).cpu().numpy()            # [N, horizon]

    # denormalise
    pred_aqhi = pred_norm * aqhi_scaler.scale_[0] + aqhi_scaler.mean_[0]
    pred_aqhi = np.clip(np.round(pred_aqhi), 1, 11).astype(int)

    result = pd.DataFrame(pred_aqhi.T, columns=DISTRICTS,
                           index=range(1, horizon + 1))
    result.index.name = "hours_ahead"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train GAGNN for HK AQHI forecasting")
    p.add_argument("--csv",        default="aqhi_history_2026.csv",
                   help="Path to CSV file or directory of CSVs (2013-2026)")
    p.add_argument("--out-dir",    default=".", type=Path,
                   help="Directory to save weights & scalers")
    p.add_argument("--epochs",     default=150,  type=int)
    p.add_argument("--batch-size", default=32,   type=int)
    p.add_argument("--seq-len",    default=24,   type=int,
                   help="Look-back window (hours)")
    p.add_argument("--horizon",    default=6,    type=int,
                   help="Forecast horizon (hours)")
    p.add_argument("--hidden-dim", default=128,  type=int)
    p.add_argument("--lr",         default=1e-3, type=float)
    p.add_argument("--patience",   default=15,   type=int,
                   help="Early stopping patience (epochs)")
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training, load best weights and evaluate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        weight_path = args.out_dir / "hk_pro_model_best.pth"
        if not weight_path.exists():
            print(f"[ERROR] No weight file at {weight_path}. Run training first.")
            sys.exit(1)
        model, fs, as_ = train.__globals__   # not used in eval-only; kept for completeness
        print("[INFO] --eval-only: load weights & run inference demo")
        df_demo = pd.read_csv(args.csv)
        model, feat_scaler, aqhi_scaler, edge_index = load_model_for_inference(
            weight_path,
            args.out_dir / "feat_scaler.pkl",
            args.out_dir / "aqhi_scaler.pkl",
        )
        result = predict(model, feat_scaler, aqhi_scaler, edge_index, df_demo)
        print("\nForecast (AQHI):")
        print(result.to_string())
    else:
        train(args)
