"""
train_model.py — GAGNN (Graph Attention Graph Neural Network)  v2.0
Hong Kong 18-District AQHI Forecasting
=========================================
WHAT'S NEW IN v2.0
  ✅ Trains THREE separate models: horizon = 3h, 6h, 24h
  ✅ Hit-Rate metric  — exact integer match rate across all (sample, station, step)
  ✅ Band Hit-Rate    — Low(1-3) / Mid(4-6) / High(7+) classification accuracy
  ✅ Cross-station MAE with automatic flag for stations >50% above mean
  ✅ Tap Mun (background) vs Causeway Bay (roadside) disparity check
  ✅ Confusion matrix per horizon (3×3 band-level)
  ✅ Edge index + weight persisted to edge_index.pt / edge_weight.pt
  ✅ Distance-based edge weighting (1/d², normalised) + mountain-barrier penalties
  ✅ Graph summary printed (top-5 closest pairs, edge count per node)
  ✅ Residual connections + LayerNorm in GAT layers
  ✅ Gradient norm tracked per epoch (detects exploding/vanishing gradients)
  ✅ Model checksum (SHA-256) printed → Render can verify loaded weights
  ✅ Loss curves saved as PNG (--plot flag)
  ✅ Circular-mean wind direction (proper vector averaging)
  ✅ Robust AQHI key matching across CSV variants

Architecture : GRU (temporal) → GAT×2 (spatial, residual) → FC (output)
Input         : past 24 hours × 18 stations × 10 features
Output        : next HORIZON hours AQHI  [B, 18, HORIZON]

CLI
  python train_model.py                           # train 3h + 6h + 24h
  python train_model.py --horizons 6              # train 6h only
  python train_model.py --csv ./data/             # directory of CSVs
  python train_model.py --eval-only --horizon 6   # load saved weights & evaluate
  python train_model.py --plot                    # also save loss curve PNGs
"""

import argparse
import hashlib
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
ALL_HORIZONS  = [3, 6, 24]
SEQ_LEN       = 24
NODE_FEATURES = 10
HIDDEN_DIM    = 128
GAT_HEADS     = 4
DROPOUT       = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STATION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
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

BARRIER_PAIRS = {
    frozenset({"Sha Tin",    "Mong Kok"}):        0.30,
    frozenset({"Sha Tin",    "Kowloon City"}):     0.30,
    frozenset({"Kwai Chung", "Central/Western"}):  0.40,
    frozenset({"Tuen Mun",   "Tsuen Wan"}):        0.50,
    frozenset({"Tung Chung", "Tsuen Wan"}):        0.45,
    frozenset({"Tap Mun",    "Sha Tin"}):          0.40,
    frozenset({"North",      "Tai Po"}):           0.60,
}

DISTRICT_HUM_MAP = {
    "Central/Western": ["HKO", "CCH"],  "Eastern":       ["JKB", "SKG"],
    "Kwun Tong":       ["TKL", "JKB"],  "Sham Shui Po":  ["HKO", "YCT"],
    "Kwai Chung":      ["KP",  "YCT"],  "Tsuen Wan":     ["KSC", "TMS"],
    "Yuen Long":       ["SHA", "LFS"],  "Tuen Mun":      ["TMS", "LFS"],
    "Tung Chung":      ["TC",  "HKA"],  "Tai Po":        ["SSH", "SEK"],
    "Sha Tin":         ["SHA", "SEK"],  "Tap Mun":       ["SKG", "SSH"],
    "Causeway Bay":    ["HKO", "JKB"],  "Central":       ["HKO", "CCH"],
    "Mong Kok":        ["HKO", "YCT"],  "Tseung Kwan O": ["TKL", "SKG"],
    "Southern":        ["HKO", "PEN"],  "North":         ["SSH", "SEK"],
}

DISTRICT_WIND_MAP = {
    "Central/Western": ["HKS", "SC"],   "Eastern":       ["JKB", "SKG"],
    "Kwun Tong":       ["TKL", "JKB"],  "Sham Shui Po":  ["CCH", "SC"],
    "Kwai Chung":      ["KP",  "NP"],   "Tsuen Wan":     ["TPK", "NP"],
    "Yuen Long":       ["SHA", "LFS"],  "Tuen Mun":      ["TME", "TUN"],
    "Tung Chung":      ["TC",  "HKA"],  "Tai Po":        ["SSH", "SEK"],
    "Sha Tin":         ["SHA", "SHL"],  "Tap Mun":       ["NGP", "WGL"],
    "Causeway Bay":    ["HKS", "SE"],   "Central":       ["HKS", "PLC"],
    "Mong Kok":        ["CCB", "CCH"],  "Tseung Kwan O": ["TKL", "SKG"],
    "Southern":        ["WLP", "WGL"],  "North":         ["SSH", "SEK"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
def _haversine_km(lat1, lon1, lat2, lon2):
    R  = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def build_graph(districts=DISTRICTS, coords=DISTRICT_COORDS,
                barrier_pairs=BARRIER_PAIRS, max_dist_km=40.0):
    """
    Build spatial graph with inverse-distance-squared weights and barrier penalties.

    Returns
    -------
    edge_index  : LongTensor  [2, E]
    edge_weight : FloatTensor [E]   (normalised to [0, 1])
    dist_matrix : ndarray     [N, N]  km inter-district distances
    """
    n = len(districts)
    src, dst, wts = [], [], []
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        lat1, lon1 = coords[districts[i]]
        for j in range(n):
            if i == j:
                continue
            lat2, lon2 = coords[districts[j]]
            d = _haversine_km(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = d
            if d > max_dist_km:
                continue
            w    = 1.0 / (d ** 2 + 1e-6)
            pair = frozenset({districts[i], districts[j]})
            w   *= barrier_pairs.get(pair, 1.0)
            src.append(i); dst.append(j); wts.append(w)

    wts_arr = np.array(wts, dtype=np.float32)
    if len(wts_arr) > 0:
        wts_arr /= wts_arr.max()          # normalise weights to [0, 1]

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts_arr,    dtype=torch.float)
    return edge_index, edge_weight, dist_matrix


def save_graph(edge_index: torch.Tensor, edge_weight: torch.Tensor, out_dir: Path):
    """
    Persist graph to .pt files so Render loads the EXACT topology
    the model was trained with — no re-computation risk.
    """
    torch.save(edge_index,  out_dir / "edge_index.pt")
    torch.save(edge_weight, out_dir / "edge_weight.pt")
    print(f"[Graph] Saved  edge_index.pt  ({edge_index.shape[1]} directed edges)")
    print(f"[Graph] Saved  edge_weight.pt "
          f"(range {edge_weight.min():.4f} – {edge_weight.max():.4f})")


def print_graph_summary(edge_index, edge_weight, dist_matrix):
    n = len(DISTRICTS)
    E = edge_index.shape[1]
    print(f"\n{'─'*58}")
    print(f"  Graph Summary")
    print(f"  Nodes            : {n}")
    print(f"  Directed edges   : {E}  (~{E//n:.0f} per node on average)")
    print(f"  Weight range     : {edge_weight.min():.4f} – {edge_weight.max():.4f}")
    print(f"\n  Top-5 closest district pairs (km):")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] > 0:
                pairs.append((dist_matrix[i, j], DISTRICTS[i], DISTRICTS[j]))
    for d, a, b in sorted(pairs)[:5]:
        print(f"    {a:<20s} ↔ {b:<20s}  {d:.2f} km")
    print(f"\n  Barrier-penalised pairs:")
    for pair, mult in BARRIER_PAIRS.items():
        names = list(pair)
        print(f"    {names[0]:<20s} ↔ {names[1]:<20s}  ×{mult}")
    print(f"{'─'*58}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def _station_mean(df: pd.DataFrame, prefix: str, stations: list) -> pd.Series:
    cols = [f"{prefix}{s}" for s in stations if f"{prefix}{s}" in df.columns]
    return df[cols].mean(axis=1) if cols else pd.Series(np.nan, index=df.index)


def engineer_features(df: pd.DataFrame):
    """
    Convert a raw CSV DataFrame to [T, N, 10] feature array + [T, N] target.

    Feature index
    -------------
    0  aqhi        current AQHI (clipped 1–11)
    1  humidity    mean of mapped HUM_* stations
    2  wspd        mean wind speed km/h
    3  wdir_sin    sin of circular-mean wind direction
    4  wdir_cos    cos of circular-mean wind direction
    5  cyclone     Cyclone_Present flag
    6  hour_sin    sin(2π·hour/24)
    7  hour_cos    cos(2π·hour/24)
    8  dow_sin     sin(2π·weekday/7)
    9  dow_cos     cos(2π·weekday/7)
    """
    df = df.copy()
  
    # 更彈性的日期解析（支援有秒 / 無秒的格式）
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce')    
    df = df.sort_values("Date").reset_index(drop=True)

    # Cleaning
    aqhi_cols = [c for c in df.columns if c.startswith("AQHI_")]
    if aqhi_cols:
        df[aqhi_cols] = df[aqhi_cols].clip(1, 11)
    hum_cols = [c for c in df.columns if c.startswith("HUM_")]
    if hum_cols:
        df[hum_cols] = df[hum_cols].clip(0, 100)
    df = df.ffill().bfill()

    hour    = df["Date"].dt.hour.values
    dow     = df["Date"].dt.dayofweek.values
    h_sin   = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    h_cos   = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    d_sin   = np.sin(2 * np.pi * dow  /  7).astype(np.float32)
    d_cos   = np.cos(2 * np.pi * dow  /  7).astype(np.float32)
    cyclone = (df["Cyclone_Present"].values.astype(np.float32)
               if "Cyclone_Present" in df.columns
               else np.zeros(len(df), dtype=np.float32))

    T = len(df)
    X = np.zeros((T, len(DISTRICTS), NODE_FEATURES), dtype=np.float32)
    y = np.zeros((T, len(DISTRICTS)),                dtype=np.float32)

    for ni, district in enumerate(DISTRICTS):
        # Robust AQHI key lookup
        aqhi_key = next(
            (k for k in [f"AQHI_{district}",
                          f"AQHI_{district.replace('/','_')}",
                          f"AQHI_{district.replace(' ','_')}"]
             if k in df.columns), None)
        aqhi = (df[aqhi_key].values.astype(np.float32)
                if aqhi_key else np.full(T, 3.0, np.float32))

        hum      = _station_mean(df, "HUM_",  DISTRICT_HUM_MAP[district]).values.astype(np.float32)
        wspd     = _station_mean(df, "WSPD_", DISTRICT_WIND_MAP[district]).values.astype(np.float32)
        wdir_raw = _station_mean(df, "PDIR_", DISTRICT_WIND_MAP[district]).values.astype(np.float32)

        # Circular-mean wind direction (correct for 350° ≈ 10°)
        sin_v = np.sin(np.deg2rad(wdir_raw))
        cos_v = np.cos(np.deg2rad(wdir_raw))
        wdir_circ = np.degrees(np.arctan2(sin_v, cos_v))
        wdir_circ = ((wdir_circ % 360) + 360) % 360
        wdir_rad  = np.deg2rad(wdir_circ)

        X[:, ni, 0] = aqhi
        X[:, ni, 1] = np.nan_to_num(hum,  nan=70.0)
        X[:, ni, 2] = np.nan_to_num(wspd, nan=10.0)
        X[:, ni, 3] = np.sin(wdir_rad)
        X[:, ni, 4] = np.cos(wdir_rad)
        X[:, ni, 5] = cyclone
        X[:, ni, 6] = h_sin
        X[:, ni, 7] = h_cos
        X[:, ni, 8] = d_sin
        X[:, ni, 9] = d_cos
        y[:, ni]    = aqhi

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class AQHIDataset(torch.utils.data.Dataset):
    def __init__(self, X_norm, y_norm, seq_len=SEQ_LEN, horizon=6):
        self.X       = X_norm
        self.y       = y_norm
        self.seq_len = seq_len
        self.horizon = horizon
        self.valid   = max(len(X_norm) - seq_len - horizon + 1, 0)

    def __len__(self):
        return self.valid

    def __getitem__(self, idx):
        x     = torch.tensor(self.X[idx:idx+self.seq_len],
                              dtype=torch.float32).permute(1, 0, 2)          # [N, T, F]
        y_tgt = torch.tensor(
            self.y[idx+self.seq_len : idx+self.seq_len+self.horizon],
            dtype=torch.float32).T                                            # [N, H]
        return x, y_tgt


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL  (Residual GAT + LayerNorm)
# ══════════════════════════════════════════════════════════════════════════════
class GAGNN(nn.Module):
    """
    GRU temporal encoder → 2× GAT spatial layers (residual + LayerNorm) → FC output.
    Input  : [B, N, T, F]
    Output : [B, N, horizon]
    """
    def __init__(self, node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 seq_len=SEQ_LEN, horizon=6, gat_heads=GAT_HEADS, dropout=DROPOUT):
        super().__init__()
        self.horizon = horizon

        self.gru  = nn.GRU(input_size=node_features, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=gat_heads,
                             concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=gat_heads,
                             concat=False, dropout=dropout)
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.ln2  = nn.LayerNorm(hidden_dim)

        self.fc   = nn.Linear(hidden_dim, horizon)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        B, N, T, F = x.shape
        dev = x.device

        # Batched edge index
        offsets    = (torch.arange(B, device=dev) * N).unsqueeze(1)
        batched_ei = (edge_index.unsqueeze(0) + offsets.unsqueeze(2)).view(2, -1)

        # Temporal phase: GRU
        gru_out, _ = self.gru(x.reshape(B * N, T, F))
        h = gru_out[:, -1, :]                             # [B*N, H]

        # Spatial phase: GAT 1 + residual
        h = self.ln1(self.relu(self.gat1(h, batched_ei)) + h)
        h = self.drop(h)
        # Spatial phase: GAT 2 + residual
        h = self.ln2(self.relu(self.gat2(h, batched_ei)) + h)

        return self.fc(h).view(B, N, self.horizon)


def model_checksum(model: nn.Module) -> str:
    """SHA-256 of all weight bytes (first 16 hex chars). Used to verify Render loaded correctly."""
    sha = hashlib.sha256()
    for p in model.parameters():
        sha.update(p.data.cpu().numpy().tobytes())
    return sha.hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
def fit_scalers(X, y):
    T, N, F = X.shape
    fs = StandardScaler().fit(X.reshape(-1, F))
    ys = StandardScaler().fit(y.reshape(-1, 1))
    return fs, ys


def apply_scalers(X, y, fs, ys):
    T, N, F = X.shape
    Xn = fs.transform(X.reshape(-1, F)).reshape(T, N, F).astype(np.float32)
    yn = ys.transform(y.reshape(-1, 1)).reshape(T, N).astype(np.float32)
    return Xn, yn


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_hit_rate(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Exact integer hit rate across all (sample, station, step) triples.

    Parameters
    ----------
    preds, targets : [S, N, H] denormalised floats

    Returns
    -------
    float in [0, 1]
    """
    p = np.clip(np.round(preds).astype(int),   1, 11)
    t = np.clip(np.round(targets).astype(int),  1, 11)
    rate = float((p == t).mean())
    print(f"  Hit Rate (exact match) : {rate*100:.2f}%  "
          f"({int((p == t).sum())} / {p.size} triples correct)")
    return rate


def compute_band_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Band accuracy: Low(1-3) / Mid(4-6) / High(7-11).
    """
    def to_band(v):
        v = np.clip(np.round(v).astype(int), 1, 11)
        return np.where(v <= 3, 0, np.where(v <= 6, 1, 2))

    pb  = to_band(preds)
    tb  = to_band(targets)
    acc = float((pb == tb).mean())
    print(f"  Band Accuracy (Low/Mid/High) : {acc*100:.2f}%")

    # Per-band breakdown
    for band_id, name in enumerate(["Low (1-3)", "Mid (4-6)", "High (7+)"]):
        mask = (tb == band_id)
        if mask.sum() > 0:
            band_acc = float((pb[mask] == tb[mask]).mean())
            count    = int(mask.sum())
            print(f"    {name:<12s} : {band_acc*100:.2f}%  (n={count})")
    return acc


def cross_station_report(preds: np.ndarray, targets: np.ndarray, horizon: int) -> dict:
    """
    Per-district MAE on the last step of the horizon.
    Flags districts >50% above mean MAE.
    Highlights Tap Mun (background) vs Causeway Bay (roadside) ratio.

    Parameters
    ----------
    preds, targets : [S, N, H] denormalised

    Returns dict {district: {"mae": float, "flag": "OK"|"HIGH"}}
    """
    p   = preds[:, :, -1]     # [S, N]  — last forecast step
    t   = targets[:, :, -1]
    mae = np.abs(p - t).mean(axis=0)   # [N]

    mean_mae  = mae.mean()
    threshold = mean_mae * 1.5

    print(f"\n  {'─'*60}")
    print(f"  Cross-Station MAE Report  (horizon +{horizon}h, last step)")
    print(f"  {'District':<22s}  {'MAE':>7s}  {'vs Mean':>8s}  Status")
    print(f"  {'─'*60}")

    report = {}
    for ni, d in enumerate(DISTRICTS):
        flag     = "⚠️  HIGH" if mae[ni] > threshold else "✅ OK"
        vs_mean  = mae[ni] / mean_mae
        print(f"  {d:<22s}  {mae[ni]:>7.3f}  {vs_mean:>7.2f}×   {flag}")
        report[d] = {"mae": float(mae[ni]),
                     "flag": "HIGH" if mae[ni] > threshold else "OK"}

    print(f"  {'─'*60}")
    print(f"  Mean MAE : {mean_mae:.3f}   Threshold (×1.5) : {threshold:.3f}")

    # Tap Mun vs Causeway Bay special check
    tm_idx = DISTRICTS.index("Tap Mun")
    cb_idx = DISTRICTS.index("Causeway Bay")
    ratio  = mae[tm_idx] / (mae[cb_idx] + 1e-8)
    verdict = ("⚠️  Background/Roadside disparity detected (Tap Mun >>  Causeway Bay)"
               if ratio > 1.5 else "✅ Background/Roadside balance is acceptable")
    print(f"\n  Tap Mun (background) MAE    : {mae[tm_idx]:.3f}")
    print(f"  Causeway Bay (roadside) MAE : {mae[cb_idx]:.3f}   Ratio: {ratio:.2f}")
    print(f"  {verdict}")

    return report


def print_confusion_matrix(preds: np.ndarray, targets: np.ndarray, horizon: int):
    """3×3 confusion matrix for Low / Mid / High AQHI bands."""
    def to_band(v):
        v = np.clip(np.round(v).astype(int), 1, 11)
        return np.where(v <= 3, 0, np.where(v <= 6, 1, 2))

    p_b  = to_band(preds[:, :, -1]).ravel()
    t_b  = to_band(targets[:, :, -1]).ravel()
    cm   = confusion_matrix(t_b, p_b, labels=[0, 1, 2])
    lbls = ["Low", "Mid", "High"]

    print(f"\n  Confusion Matrix  (horizon +{horizon}h, last step)")
    header = "  {:12s}".format("True \\ Pred") + "".join(f"  {l:>8s}" for l in lbls)
    print(header)
    print(f"  {'─'*45}")
    for i, row in enumerate(cm):
        line = f"  {lbls[i]:<12s}" + "".join(f"  {v:>8d}" for v in row)
        print(line)

    # Recall per band (diagonal / row sum)
    print(f"\n  Per-band recall:")
    for i, l in enumerate(lbls):
        total = cm[i].sum()
        rec   = cm[i, i] / total if total > 0 else 0.0
        print(f"    {l:<5s} : {rec*100:.1f}%  (n={total})")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SINGLE-HORIZON TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_one_horizon(horizon, X_tr_n, y_tr_n, X_va_n, y_va_n, X_te_n, y_te_n,
                      feat_scaler, aqhi_scaler, edge_index, args, device):
    print(f"\n{'═'*60}")
    print(f"  TRAINING  horizon = +{horizon}h")
    print(f"{'═'*60}")

    ds_tr = AQHIDataset(X_tr_n, y_tr_n, SEQ_LEN, horizon)
    ds_va = AQHIDataset(X_va_n, y_va_n, SEQ_LEN, horizon)
    ds_te = AQHIDataset(X_te_n, y_te_n, SEQ_LEN, horizon)
    print(f"[Data] train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")

    if len(ds_tr) == 0:
        print(f"[SKIP] Not enough data for horizon={horizon}h. Skipping.")
        return {}

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, shuffle=True,  **dl_kwargs)
    dl_va = torch.utils.data.DataLoader(ds_va, shuffle=False, **dl_kwargs)
    dl_te = torch.utils.data.DataLoader(ds_te, shuffle=False, **dl_kwargs)

    model = GAGNN(node_features=NODE_FEATURES, hidden_dim=args.hidden_dim,
                  seq_len=SEQ_LEN, horizon=horizon).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    criterion = nn.L1Loss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_val   = float("inf")
    patience_c = 0
    scale      = aqhi_scaler.scale_[0]
    best_path  = args.out_dir / f"hk_pro_model_best_{horizon}h.pth"
    # 6h is the canonical backward-compat weight for gnn_main.py
    canonical  = args.out_dir / "hk_pro_model_best.pth"

    edge_idx   = edge_index.to(device)
    tr_losses, va_losses = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss, total_gnorm = 0.0, 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb, edge_idx), yb)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_gnorm += float(gnorm)
            optimiser.step()
            tr_loss += loss.item()
        tr_loss /= len(dl_tr)
        avg_gnorm = total_gnorm / len(dl_tr)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                va_loss += criterion(model(xb.to(device), edge_idx),
                                     yb.to(device)).item()
        va_loss /= len(dl_va)
        scheduler.step(va_loss)

        tr_losses.append(tr_loss * scale)
        va_losses.append(va_loss * scale)
        lr_now = optimiser.param_groups[0]["lr"]

        print(f"  Ep {epoch:04d}/{args.epochs}  "
              f"Tr={tr_loss*scale:.4f}  Va={va_loss*scale:.4f}  "
              f"GNorm={avg_gnorm:.3f}  lr={lr_now:.1e}  {time.time()-t0:.1f}s")

        if va_loss < best_val:
            best_val   = va_loss
            patience_c = 0
            torch.save(model.state_dict(), best_path)
            if horizon == 6:
                torch.save(model.state_dict(), canonical)
            print(f"  [✅ Checkpoint] val={va_loss*scale:.4f}  → {best_path.name}")
        else:
            patience_c += 1
            if patience_c >= args.patience:
                print(f"  [EarlyStop] epoch={epoch}  "
                      f"no improvement for {args.patience} epochs")
                break

    if args.plot:
        _save_loss_curve(tr_losses, va_losses, horizon, args.out_dir)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n[Test] Loading {best_path.name} …")
    model.load_state_dict(torch.load(best_path, map_location=device))
    chk = model_checksum(model)
    print(f"[Test] Model checksum : {chk}")
    model.eval()

    te_loss = 0.0
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            pred = model(xb.to(device), edge_idx)
            te_loss += criterion(pred, yb.to(device)).item()
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(yb.numpy())
    te_loss /= len(dl_te)

    preds_n   = np.concatenate(all_pred, 0)   # [S, N, H]
    targets_n = np.concatenate(all_tgt,  0)

    preds_r   = preds_n   * scale + aqhi_scaler.mean_[0]
    targets_r = targets_n * scale + aqhi_scaler.mean_[0]

    print(f"\n{'─'*60}")
    print(f"  Test Results  (horizon +{horizon}h)")
    print(f"  MAE (AQHI scale) : {te_loss*scale:.4f}")

    hit  = compute_hit_rate(preds_r, targets_r)
    bacc = compute_band_accuracy(preds_r, targets_r)
    cross_station_report(preds_r, targets_r, horizon)
    print_confusion_matrix(preds_r, targets_r, horizon)

    return {
        "horizon":       horizon,
        "test_mae":      float(te_loss * scale),
        "hit_rate":      hit,
        "band_accuracy": bacc,
        "checksum":      chk,
        "best_path":     str(best_path),
    }


def _save_loss_curve(tr_losses, va_losses, horizon, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ep = range(1, len(tr_losses)+1)
        ax.plot(ep, tr_losses, label="Train MAE")
        ax.plot(ep, va_losses, label="Val MAE")
        ax.set(xlabel="Epoch", ylabel="MAE (AQHI)", title=f"Loss Curve  +{horizon}h")
        ax.legend(); fig.tight_layout()
        path = out_dir / f"loss_curve_{horizon}h.png"
        fig.savefig(path, dpi=120); plt.close()
        print(f"[Plot] Loss curve → {path}")
    except ImportError:
        print("[Plot] matplotlib not installed — skipping")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device   : {device}")
    print(f"[INFO] Horizons : {args.horizons}")
    print(f"[INFO] SeqLen   : {SEQ_LEN}  |  NodeFeatures: {NODE_FEATURES}")

    # ── Load data ─────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if csv_path.is_dir():
        dfs = [pd.read_csv(p) for p in sorted(csv_path.glob("*.csv"))]
        df  = pd.concat(dfs, ignore_index=True)
        print(f"[CSV] Loaded {len(dfs)} files → {len(df)} rows total")
    else:
        df = pd.read_csv(csv_path)
        print(f"[CSV] Loaded {len(df)} rows from {csv_path}")

    # ── Feature engineering ───────────────────────────────────────────────────
    print("[Features] Engineering …")
    X, y = engineer_features(df)
    print(f"[Features] X={X.shape}  y={y.shape}")

    # ── Chronological 80/10/10 split ──────────────────────────────────────────
    T  = len(X)
    t1 = int(T * 0.80); t2 = int(T * 0.90)
    X_tr, y_tr = X[:t1],   y[:t1]
    X_va, y_va = X[t1:t2], y[t1:t2]
    X_te, y_te = X[t2:],   y[t2:]
    print(f"[Split] train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    fs, ys = fit_scalers(X_tr, y_tr)
    joblib.dump(fs, args.out_dir / "feat_scaler.pkl")
    joblib.dump(ys, args.out_dir / "aqhi_scaler.pkl")
    print(f"[Scalers] feat_scaler.pkl + aqhi_scaler.pkl saved")
    print(f"[Scalers] AQHI mean={ys.mean_[0]:.3f}  scale={ys.scale_[0]:.3f}")

    X_tr_n, y_tr_n = apply_scalers(X_tr, y_tr, fs, ys)
    X_va_n, y_va_n = apply_scalers(X_va, y_va, fs, ys)
    X_te_n, y_te_n = apply_scalers(X_te, y_te, fs, ys)

    # ── High-risk oversampling (AQHI ≥ 8) ─────────────────────────────────────
    mask     = (y_tr >= 8).any(axis=1)
    hi_count = int(mask.sum())
    if hi_count > 0:
        rep    = max(1, int(len(X_tr_n) * 0.15 / hi_count))
        X_tr_n = np.concatenate([X_tr_n, np.repeat(X_tr_n[mask], rep, 0)])
        y_tr_n = np.concatenate([y_tr_n, np.repeat(y_tr_n[mask], rep, 0)])
        print(f"[Oversample] {hi_count} high-risk rows ×{rep} → {len(X_tr_n)} total")

    # ── Graph ─────────────────────────────────────────────────────────────────
    print("[Graph] Building …")
    edge_index, edge_weight, dist_matrix = build_graph()
    print_graph_summary(edge_index, edge_weight, dist_matrix)
    save_graph(edge_index, edge_weight, args.out_dir)

    # ── Train each horizon ─────────────────────────────────────────────────────
    reports = []
    for h in args.horizons:
        r = train_one_horizon(h, X_tr_n, y_tr_n, X_va_n, y_va_n, X_te_n, y_te_n,
                              fs, ys, edge_index, args, device)
        reports.append(r)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  FINAL SUMMARY")
    print(f"  {'Horizon':<10s}  {'MAE':>8s}  {'HitRate':>9s}  {'BandAcc':>9s}  Checksum")
    print(f"  {'─'*58}")
    for r in reports:
        if r:
            print(f"  +{r['horizon']:<9d}  {r['test_mae']:>8.4f}  "
                  f"{r['hit_rate']*100:>8.2f}%  {r['band_accuracy']*100:>8.2f}%  "
                  f"{r['checksum']}")
    print(f"\n  Output directory: {args.out_dir}")
    print(f"  Files: feat_scaler.pkl  aqhi_scaler.pkl  edge_index.pt  edge_weight.pt")
    for h in args.horizons:
        print(f"         hk_pro_model_best_{h}h.pth")
    print(f"{'═'*60}\n")
    return reports


# ══════════════════════════════════════════════════════════════════════════════
# 10.  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_model_for_inference(weight_path, feat_scaler_path, aqhi_scaler_path,
                              edge_index_path=None, horizon=6, device=None):
    """
    Load model + scalers + graph for production inference on Render.
    Loads persisted edge_index.pt if available (recommended).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = joblib.load(feat_scaler_path)
    ys = joblib.load(aqhi_scaler_path)

    model = GAGNN(node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                  seq_len=SEQ_LEN, horizon=horizon).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f"[Inference] Loaded +{horizon}h  checksum={model_checksum(model)}")

    if edge_index_path and Path(edge_index_path).exists():
        ei = torch.load(edge_index_path, map_location=device)
        print(f"[Inference] Loaded persisted edge_index ({ei.shape[1]} edges)")
    else:
        ei, _, _ = build_graph()
        ei = ei.to(device)
        print("[Inference] Rebuilt edge_index from coordinates")

    return model, fs, ys, ei


def predict(model, feat_scaler, aqhi_scaler, edge_index,
            recent_df: pd.DataFrame, horizon: int = 6, device=None) -> pd.DataFrame:
    """
    Run inference on recent data.

    Parameters
    ----------
    recent_df : raw CSV DataFrame  (at least SEQ_LEN rows)
    horizon   : must match the loaded model's output horizon

    Returns
    -------
    DataFrame[DISTRICTS × horizon steps]  integer AQHI values
    """
    if device is None:
        device = next(model.parameters()).device

    X, _ = engineer_features(recent_df.tail(SEQ_LEN + horizon))
    X    = X[-SEQ_LEN:]
    T, N, F = X.shape
    Xn = feat_scaler.transform(X.reshape(-1, F)).reshape(T, N, F)

    # [T, N, F] → [1, N, T, F]
    xt = torch.tensor(Xn, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0)

    with torch.no_grad():
        pn = model(xt.to(device), edge_index).squeeze(0).cpu().numpy()  # [N, H]

    pred = np.clip(np.round(pn * aqhi_scaler.scale_[0] + aqhi_scaler.mean_[0]),
                   1, 11).astype(int)
    return pd.DataFrame(pred.T, columns=DISTRICTS,
                        index=pd.RangeIndex(1, horizon+1, name="hours_ahead"))


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="GAGNN v2 — Multi-horizon HK AQHI")
    p.add_argument("--csv",        default="aqhi_history_2026.csv")
    p.add_argument("--out-dir",    default=".", type=Path)
    p.add_argument("--horizons",   default=None, nargs="+", type=int,
                   help="Horizons to train (default: 3 6 24)")
    p.add_argument("--epochs",     default=150, type=int)
    p.add_argument("--batch-size", default=32,  type=int)
    p.add_argument("--hidden-dim", default=HIDDEN_DIM, type=int)
    p.add_argument("--lr",         default=1e-3, type=float)
    p.add_argument("--patience",   default=15,   type=int)
    p.add_argument("--plot",       action="store_true",
                   help="Save loss-curve PNGs")
    p.add_argument("--eval-only",  action="store_true",
                   help="Load saved weights and evaluate only")
    p.add_argument("--horizon",    default=6, type=int,
                   help="Single horizon for --eval-only mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.horizons is None:
        args.horizons = ALL_HORIZONS

    if args.eval_only:
        h  = args.horizon
        wp = args.out_dir / f"hk_pro_model_best_{h}h.pth"
        if not wp.exists():
            wp = args.out_dir / "hk_pro_model_best.pth"   # canonical fallback
        if not wp.exists():
            print(f"[ERROR] No weight file at {wp}. Run training first.")
            sys.exit(1)

        model, fs, ys, ei = load_model_for_inference(
            weight_path      = wp,
            feat_scaler_path = args.out_dir / "feat_scaler.pkl",
            aqhi_scaler_path = args.out_dir / "aqhi_scaler.pkl",
            edge_index_path  = args.out_dir / "edge_index.pt",
            horizon          = h,
        )
        csv_file = args.csv if not Path(args.csv).is_dir() \
                   else next(Path(args.csv).glob("*.csv"))
        df = pd.read_csv(csv_file)
        result = predict(model, fs, ys, ei, df, horizon=h)
        print(f"\n  Forecast  +{h}h  (AQHI):")
        print(result.to_string())
    else:
        train(args)
