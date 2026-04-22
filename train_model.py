"""
gnn_main.py  v2.0 — BuildTech GAGNN Prediction Service
========================================================
https://buildtech-gnn-service.onrender.com/

Fully aligned with train_model.py v2.0:
  ✅ GAGNN class mirrors training exactly (Residual + LayerNorm)
  ✅ Loads persisted edge_index.pt / edge_weight.pt (not rebuilt)
  ✅ SHA-256 checksum verification on every model load
  ✅ Label sanitiser: Firebase "Central/Western" → "Central_Western" → AQHI lookup
  ✅ Circular-mean wind direction (vector averaging, not arithmetic)
  ✅ Full hour + day-of-week sin/cos temporal encoding
  ✅ Input tensor sanity check before inference (NaN / OOB guard)
  ✅ Horizon dispatcher: 3h / 6h / 24h model loop
  ✅ Post-processing: clip [1–11] + round to int
  ✅ Firebase auto-retry (3 attempts, exponential back-off)
  ✅ Dual CSV: today-overwrite + history-append
  ✅ Inference latency tracking per horizon
  ✅ Model drift detection (warns if avg error > 3 AQHI levels, 3 runs)
  ✅ Rich HTML dashboard (system status, per-district table, latency, drift)

Endpoints
  GET  /                   HTML dashboard (auto-refresh 5 min)
  GET  /status             JSON health check
  POST /sync               Receive hourly record from GitHub Action
  POST /predict            Manually trigger pipeline
  GET  /predictions        Latest prediction JSON (3h / 6h / 24h)
  GET  /predictions/csv    Download today's CSV
  GET  /history            Rolling prediction log JSON
  GET  /history/csv        Download full history CSV
  GET  /firebase/test      Ping Firebase read/write
  GET  /metrics            Inference latency + drift report JSON

Render environment variables required
  FIREBASE_SERVICE_ACCOUNT_JSON  — full JSON of Firebase service account
  (optional) EXPECTED_CHECKSUM_3H / _6H / _24H  — from training log
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path


import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, Response, jsonify, render_template_string, request
from apscheduler.schedulers.background import BackgroundScheduler
from torch_geometric.nn import GATConv

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING  (stdout so Render captures it)
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("GAGNN")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS  — must mirror train_model.py exactly
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).parent
FEAT_SCALER_PATH  = BASE_DIR / "feat_scaler.pkl"
AQHI_SCALER_PATH  = BASE_DIR / "aqhi_scaler.pkl"
EDGE_INDEX_PATH   = BASE_DIR / "edge_index.pt"
EDGE_WEIGHT_PATH  = BASE_DIR / "edge_weight.pt"
CSV_TODAY_PATH    = BASE_DIR / "gagnn_prediction_today.csv"
CSV_HISTORY_PATH  = BASE_DIR / "gagnn_prediction_history.csv"

# Per-horizon model weight files produced by train_model.py v2
MODEL_PATHS = {
    3:  BASE_DIR / "hk_pro_model_best_3h.pth",
    6:  BASE_DIR / "hk_pro_model_best_6h.pth",
    24: BASE_DIR / "hk_pro_model_best_24h.pth",
}
# Fallback: single canonical weight if per-horizon files absent
MODEL_CANONICAL = BASE_DIR / "hk_pro_model_best.pth"

FIREBASE_DB_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app"

# Hyper-params — must match training
NODE_FEATURES = 10
HIDDEN_DIM    = 128
GAT_HEADS     = 4
DROPOUT       = 0.2
SEQ_LEN       = 24
HORIZONS      = [3, 6, 24]
N_STATIONS    = 18

# ══════════════════════════════════════════════════════════════════════════════
# DISTRICT DEFINITIONS  — identical to train_model.py
# ══════════════════════════════════════════════════════════════════════════════
DISTRICTS = [
    "Central/Western", "Eastern",      "Kwun Tong",    "Sham Shui Po",
    "Kwai Chung",      "Tsuen Wan",    "Yuen Long",    "Tuen Mun",
    "Tung Chung",      "Tai Po",       "Sha Tin",      "Tap Mun",
    "Causeway Bay",    "Central",      "Mong Kok",     "Tseung Kwan O",
    "Southern",        "North",
]

DISTRICT_COORDS = {
    "Central/Western": (22.2859, 114.1448), "Eastern":       (22.2820, 114.2210),
    "Kwun Tong":       (22.3130, 114.2240), "Sham Shui Po":  (22.3303, 114.1628),
    "Kwai Chung":      (22.3540, 114.1290), "Tsuen Wan":     (22.3710, 114.1140),
    "Yuen Long":       (22.4450, 114.0220), "Tuen Mun":      (22.3910, 113.9730),
    "Tung Chung":      (22.2890, 113.9430), "Tai Po":        (22.4510, 114.1650),
    "Sha Tin":         (22.3850, 114.1880), "Tap Mun":       (22.4710, 114.3610),
    "Causeway Bay":    (22.2800, 114.1840), "Central":       (22.2820, 114.1580),
    "Mong Kok":        (22.3200, 114.1690), "Tseung Kwan O": (22.3070, 114.2590),
    "Southern":        (22.2470, 114.1580), "North":         (22.4960, 114.1380),
}

# ── Firebase key sanitiser (解決 Central/Western 中的 / 問題) ───────────────
def sanitize_firebase_key(district: str) -> str:
    """把區域名稱轉成 Firebase 安全的 key（把 / 和空格換成 _）"""
    return district.replace('/', '_').replace(' ', '_')

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
    "Central/Western": ["HKO","CCH"], "Eastern":       ["JKB","SKG"],
    "Kwun Tong":       ["TKL","JKB"], "Sham Shui Po":  ["HKO","YCT"],
    "Kwai Chung":      ["KP", "YCT"], "Tsuen Wan":     ["KSC","TMS"],
    "Yuen Long":       ["SHA","LFS"], "Tuen Mun":      ["TMS","LFS"],
    "Tung Chung":      ["TC", "HKA"], "Tai Po":        ["SSH","SEK"],
    "Sha Tin":         ["SHA","SEK"], "Tap Mun":       ["SKG","SSH"],
    "Causeway Bay":    ["HKO","JKB"], "Central":       ["HKO","CCH"],
    "Mong Kok":        ["HKO","YCT"], "Tseung Kwan O": ["TKL","SKG"],
    "Southern":        ["HKO","PEN"], "North":         ["SSH","SEK"],
}

DISTRICT_WIND_MAP = {
    "Central/Western": ["HKS","SC"],  "Eastern":       ["JKB","SKG"],
    "Kwun Tong":       ["TKL","JKB"], "Sham Shui Po":  ["CCH","SC"],
    "Kwai Chung":      ["KP", "NP"],  "Tsuen Wan":     ["TPK","NP"],
    "Yuen Long":       ["SHA","LFS"], "Tuen Mun":      ["TME","TUN"],
    "Tung Chung":      ["TC", "HKA"], "Tai Po":        ["SSH","SEK"],
    "Sha Tin":         ["SHA","SHL"], "Tap Mun":       ["NGP","WGL"],
    "Causeway Bay":    ["HKS","SE"],  "Central":       ["HKS","PLC"],
    "Mong Kok":        ["CCB","CCH"], "Tseung Kwan O": ["TKL","SKG"],
    "Southern":        ["WLP","WGL"], "North":         ["SSH","SEK"],
}

# Dashboard key-map: DISTRICTS → GAGNN_v2 Firebase keys (for HTML compat)
V2_KEY_MAP = {
    "Causeway Bay":    "Causeway_Bay_Roadside",
    "Central":         "Central_Roadside",
    "Central/Western": "Central_Western_General",
    "Eastern":         "Eastern_General",
    "Kwai Chung":      "Kwai_Chung_General",
    "Kwun Tong":       "Kwun_Tong_General",
    "Mong Kok":        "Mong_Kok_Roadside",
    "North":           "North_General",
    "Sha Tin":         "Sha_Tin_General",
    "Sham Shui Po":    "Sham_Shui_Po_General",
    "Southern":        "Southern_General",
    "Tai Po":          "Tai_Po_General",
    "Tap Mun":         "Tap_Mun_General",
    "Tseung Kwan O":   "Tseung_Kwan_O_General",
    "Tsuen Wan":       "Tsuen_Wan_General",
    "Tuen Mun":        "Tuen_Mun_General",
    "Tung Chung":      "Tung_Chung_General",
    "Yuen Long":       "Yuen_Long_General",
}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — GAGNN MODEL CLASS  (exact mirror of train_model.py v2.0)
# ══════════════════════════════════════════════════════════════════════════════
class GAGNN(nn.Module):
    """
    GRU temporal encoder → 2× GAT spatial layers (Residual + LayerNorm) → FC.
    Input  : [B, N, T, F]
    Output : [B, N, horizon]
    CRITICAL: This class definition must stay byte-for-byte identical to
              train_model.py so torch.load_state_dict() succeeds.
    """
    def __init__(self, node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 seq_len=SEQ_LEN, horizon=6, gat_heads=GAT_HEADS, dropout=DROPOUT):
        super().__init__()
        self.horizon = horizon

        # ── Temporal ──────────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=node_features, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )

        # ── Spatial (GAT + residual + LayerNorm) ──────────────────────────────
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=gat_heads,
                             concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=gat_heads,
                             concat=False, dropout=dropout)
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.ln2  = nn.LayerNorm(hidden_dim)

        # ── Output ────────────────────────────────────────────────────────────
        self.fc   = nn.Linear(hidden_dim, horizon)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """x: [B, N, T, F]"""
        B, N, T, F = x.shape
        dev = x.device

        # Batched edge index: offset each graph in the batch
        offsets    = (torch.arange(B, device=dev) * N).unsqueeze(1)         # [B, 1]
        batched_ei = (edge_index.unsqueeze(0) + offsets.unsqueeze(2)).view(2, -1)

        # GRU temporal encoding
        gru_out, _ = self.gru(x.reshape(B * N, T, F))
        h = gru_out[:, -1, :]                                                # [B*N, H]

        # GAT spatial layer 1 + residual
        h = self.ln1(self.relu(self.gat1(h, batched_ei)) + h)
        h = self.drop(h)
        # GAT spatial layer 2 + residual
        h = self.ln2(self.relu(self.gat2(h, batched_ei)) + h)

        return self.fc(h).view(B, N, self.horizon)                           # [B, N, H]


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1a — CHECKSUM VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_checksum(model: nn.Module) -> str:
    """SHA-256 of all parameter bytes → first 16 hex chars."""
    sha = hashlib.sha256()
    for p in model.parameters():
        sha.update(p.data.cpu().numpy().tobytes())
    return sha.hexdigest()[:16]


def verify_checksum(model: nn.Module, horizon: int) -> bool:
    """
    Compare model checksum against EXPECTED_CHECKSUM_<H>H env var.
    Logs a WARNING if mismatch, but never blocks inference.
    """
    actual   = compute_checksum(model)
    env_key  = f"EXPECTED_CHECKSUM_{horizon}H"
    expected = os.environ.get(env_key, "").strip()

    log.info(f"[Checksum] horizon={horizon}h  actual={actual}  "
             f"expected={'(not set)' if not expected else expected}")

    if expected and actual != expected:
        log.warning(f"[Checksum] ⚠️  MISMATCH for {horizon}h! "
                    f"Loaded weights may differ from training log. "
                    f"Set env var {env_key}={actual} to suppress this warning.")
        return False
    if expected and actual == expected:
        log.info(f"[Checksum] ✅ Verified {horizon}h model matches training checksum")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1b — GRAPH PERSISTENCE LOADER
# ══════════════════════════════════════════════════════════════════════════════
def _haversine_km(lat1, lon1, lat2, lon2):
    R  = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def load_or_build_graph(dev: torch.device):
    """
    Priority:
      1. Load edge_index.pt + edge_weight.pt saved by train_model.py → exact match
      2. Rebuild from coordinates as fallback (no barrier weights in .pt means training used rebuild)

    Returns edge_index [2, E] and edge_weight [E] on the target device.
    """
    if EDGE_INDEX_PATH.exists() and EDGE_WEIGHT_PATH.exists():
        ei = torch.load(EDGE_INDEX_PATH,  map_location=dev)
        ew = torch.load(EDGE_WEIGHT_PATH, map_location=dev)
        log.info(f"[Graph] ✅ Loaded persisted edge_index.pt  ({ei.shape[1]} edges)")
        log.info(f"[Graph]    edge_weight range: {ew.min():.4f} – {ew.max():.4f}")
        return ei, ew
    else:
        log.warning("[Graph] edge_index.pt not found — rebuilding from coordinates")
        log.warning("[Graph] ⚠️  Barrier-penalty weights may differ from training!")
        n = len(DISTRICTS)
        src, dst, wts = [], [], []
        for i in range(n):
            lat1, lon1 = DISTRICT_COORDS[DISTRICTS[i]]
            for j in range(n):
                if i == j: continue
                lat2, lon2 = DISTRICT_COORDS[DISTRICTS[j]]
                d = _haversine_km(lat1, lon1, lat2, lon2)
                if d > 40.0: continue
                w    = 1.0 / (d**2 + 1e-6)
                pair = frozenset({DISTRICTS[i], DISTRICTS[j]})
                w   *= BARRIER_PAIRS.get(pair, 1.0)
                src.append(i); dst.append(j); wts.append(w)
        wts_arr = np.array(wts, dtype=np.float32)
        if len(wts_arr): wts_arr /= wts_arr.max()
        ei = torch.tensor([src, dst], dtype=torch.long).to(dev)
        ew = torch.tensor(wts_arr, dtype=torch.float).to(dev)
        log.info(f"[Graph] Built from scratch: {ei.shape[1]} directed edges")
        return ei, ew


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FEATURE PIPELINE (aligned with train_model.py v2.0)
# ══════════════════════════════════════════════════════════════════════════════

def sanitise_label(raw_key: str) -> str:
    """
    Convert Firebase AQHI key variants → canonical DISTRICTS list lookup key.
    Examples
      "AQHI_Central/Western" → tries "Central/Western" first (exact match)
      "AQHI_Central_Western" → maps back to "Central/Western"
    Returns the matched district name or None.
    """
    # Strip prefix
    stripped = raw_key.replace("AQHI_", "", 1)
    # Exact match first
    if stripped in DISTRICTS:
        return stripped
    # Underscore → slash (Firebase can't store "/" in keys)
    candidate = stripped.replace("_", "/", 1)
    if candidate in DISTRICTS:
        return candidate
    # Underscore → space
    candidate2 = stripped.replace("_", " ")
    if candidate2 in DISTRICTS:
        return candidate2
    return None


def _stn_mean(record: dict, prefix: str, stations: list) -> float:
    """Mean of available sensor values; returns 0.0 if all missing."""
    vals = []
    for s in stations:
        v = record.get(f"{prefix}{s}")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else 0.0


def _circular_mean_deg(record: dict, prefix: str, stations: list) -> float:
    """
    Vector-average wind direction (correct for wrap-around: 350° + 10° = 0°).
    Direct arithmetic mean would give 180° — wrong!
    """
    dirs = []
    for s in stations:
        v = record.get(f"{prefix}{s}")
        if v is not None:
            dirs.append(float(v))
    if not dirs:
        return 0.0
    sin_m = np.mean([math.sin(math.radians(d)) for d in dirs])
    cos_m = np.mean([math.cos(math.radians(d)) for d in dirs])
    deg   = math.degrees(math.atan2(sin_m, cos_m))
    return (deg % 360 + 360) % 360


def record_to_node_features(record: dict, ts: datetime) -> np.ndarray:
    """
    Convert one hourly Firebase record to [N, 10] node feature array.

    Feature index (must match train_model.py exactly):
      0  aqhi       current AQHI clipped 1-11
      1  humidity   mean of mapped HUM_* stations, clipped 0-100
      2  wspd       mean wind speed km/h
      3  wdir_sin   sin(circular-mean PDIR)
      4  wdir_cos   cos(circular-mean PDIR)
      5  cyclone    Cyclone_Present 0/1
      6  hour_sin   sin(2π·hour/24)
      7  hour_cos   cos(2π·hour/24)
      8  dow_sin    sin(2π·weekday/7)
      9  dow_cos    cos(2π·weekday/7)
    """
    # Temporal encoding
    h_sin = math.sin(2 * math.pi * ts.hour / 24)
    h_cos = math.cos(2 * math.pi * ts.hour / 24)
    d_sin = math.sin(2 * math.pi * ts.weekday() / 7)
    d_cos = math.cos(2 * math.pi * ts.weekday() / 7)
    cyclone = float(record.get("Cyclone_Present", 0) or 0)

    rows = []
    for district in DISTRICTS:
        # AQHI — robust multi-variant lookup (Firebase may store with _  or /)
        aqhi = None
        for k in [
            f"AQHI_{district}",
            f"AQHI_{district.replace('/','_')}",
            f"AQHI_{district.replace('/','_').replace(' ','_')}",
            f"AQHI_{district.replace(' ','_')}",
        ]:
            if k in record:
                aqhi = float(record[k])
                break
        aqhi = float(np.clip(aqhi if aqhi is not None else 3.0, 1.0, 11.0))

        hum  = float(np.clip(_stn_mean(record, "HUM_",  DISTRICT_HUM_MAP[district]),  0.0, 100.0))
        wspd = _stn_mean(record, "WSPD_", DISTRICT_WIND_MAP[district])
        wdir = _circular_mean_deg(record, "PDIR_", DISTRICT_WIND_MAP[district])
        wrad = math.radians(wdir)

        rows.append([
            aqhi, hum, wspd,
            math.sin(wrad), math.cos(wrad),
            cyclone,
            h_sin, h_cos, d_sin, d_cos,
        ])
    return np.array(rows, dtype=np.float32)   # [N, 10]


def input_sanity_check(X_norm: np.ndarray) -> bool:
    """
    Validate normalised input tensor before inference.
    Returns True if clean, False + logs warnings if anomalies found.
    """
    ok = True
    if np.isnan(X_norm).any():
        n_nan = int(np.isnan(X_norm).sum())
        log.warning(f"[Sanity] ⚠️  {n_nan} NaN values detected in input tensor — filling with 0")
        X_norm[:] = np.nan_to_num(X_norm, nan=0.0)
        ok = False
    if np.isinf(X_norm).any():
        n_inf = int(np.isinf(X_norm).sum())
        log.warning(f"[Sanity] ⚠️  {n_inf} Inf values detected — clipping")
        X_norm[:] = np.clip(X_norm, -10.0, 10.0)
        ok = False

    # Check raw (de-normalised) humidity proxy: feature index 1
    # After StandardScaler, a humidity of 0 raw → large negative value
    # Threshold: anything < -5 std is suspicious
    hum_col = X_norm[:, :, 1]
    if (hum_col < -5.0).any():
        log.warning(f"[Sanity] ⚠️  Humidity feature has extreme low values "
                    f"(min={hum_col.min():.2f}) — possible sensor fault")
        ok = False

    if ok:
        log.info(f"[Sanity] ✅ Input clean  "
                 f"shape={X_norm.shape}  "
                 f"min={X_norm.min():.3f}  max={X_norm.max():.3f}")
    return ok


def build_feature_window(records: list) -> tuple:
    """
    Convert list of Firebase records (oldest first) → normalised [1, N, T, F] tensor.
    Returns (X_tensor, raw_X_norm) or (None, None) on failure.
    """
    if not records:
        log.error("[Feature] No records — cannot build feature window")
        return None, None

    if len(records) < SEQ_LEN:
        n_pad = SEQ_LEN - len(records)
        log.warning(f"[Feature] Only {len(records)} records, need {SEQ_LEN}. "
                    f"Padding front with {n_pad} copies of oldest record.")
        records = [records[0]] * n_pad + list(records)

    records = records[-SEQ_LEN:]
    log.info(f"[Feature] Building window: {len(records)} records  "
             f"({records[0].get('Date','?')} → {records[-1].get('Date','?')})")

    frames = []
    for rec in records:
        # Parse timestamp
        date_raw = rec.get("Date") or ""
        key_raw  = rec.get("_key", "")
        ts       = None
        for fmt_str in [date_raw, key_raw.replace("_"," ",1).replace("-",":",2)]:
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
                try:
                    ts = datetime.strptime(fmt_str[:16], fmt)
                    break
                except Exception:
                    pass
            if ts: break
        if ts is None:
            ts = datetime.now(timezone.utc)
            log.warning(f"[Feature] Could not parse timestamp from '{date_raw}' / '{key_raw}' — using now()")

        frames.append(record_to_node_features(rec, ts))   # [N, F]

    # Stack: [SEQ, N, F] → transpose → [N, SEQ, F]
    X = np.stack(frames, axis=0)   # [SEQ, N, F]
    X = X.transpose(1, 0, 2)       # [N, SEQ, F]

    # Normalise using training scaler
    N, T, F = X.shape
    X_flat  = X.reshape(-1, F)
    X_norm  = feat_scaler.transform(X_flat).reshape(N, T, F).astype(np.float32)

    # Sanity check (modifies in-place if needed)
    input_sanity_check(X_norm)

    # Add batch dim → [1, N, T, F]
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)
    log.info(f"[Feature] ✅ Tensor shape: {tuple(X_tensor.shape)}")
    return X_tensor, X_norm


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def postprocess(pred_arr: np.ndarray) -> np.ndarray:
    """
    Clip to [1, 11] and round to integer — AQHI format rule.
    Input:  [N] or [N, H] denormalised floats
    Output: same shape, int32
    """
    return np.clip(np.round(pred_arr), 1, 11).astype(np.int32)


def run_inference(X_tensor: torch.Tensor) -> dict:
    """
    Horizon dispatcher: sequentially call 3h / 6h / 24h models.
    Returns {horizon_int: {district: aqhi_int, ...}, ...}

    Each model's LAST output step is used as the point forecast
    (e.g. for the 6h model, step index 5 = exactly 6 hours ahead).
    """
    if any(m is None for m in [models[3], models[6], models[24]]):
        log.error("[Inference] ❌ One or more models not loaded — returning fallback")
        fallback = {d: 3 for d in DISTRICTS}
        return {3: fallback, 6: fallback, 24: fallback}

    X_dev    = X_tensor.to(device)
    edge_dev = edge_index.to(device)
    results  = {}

    with torch.no_grad():
        for h in HORIZONS:
            mdl = models[h]
            log.info(f"[Inference] ▶ Horizon +{h}h …")
            t0 = time.perf_counter()
            try:
                out = mdl(X_dev, edge_dev)          # [1, N, H]
                # Take last step of the horizon (exact future point)
                pred_norm = out[0, :, -1].cpu().numpy()  # [N]

                # Denormalise: inverse of StandardScaler(AQHI)
                pred_raw  = pred_norm * aqhi_scaler.scale_[0] + aqhi_scaler.mean_[0]
                pred_int  = postprocess(pred_raw)    # [N] int

                district_preds = {DISTRICTS[i]: int(pred_int[i]) for i in range(N_STATIONS)}
                results[h]     = district_preds

                elapsed_ms = (time.perf_counter() - t0) * 1000
                avg_v = float(np.mean(pred_int))
                log.info(f"[Inference] ✅ +{h}h  avg={avg_v:.2f}  "
                         f"min={pred_int.min()}  max={pred_int.max()}  "
                         f"latency={elapsed_ms:.1f}ms")
                log.info(f"[Inference]    {district_preds}")

                # Record latency
                latency_log[h].append(round(elapsed_ms, 1))
                if len(latency_log[h]) > 100:
                    latency_log[h].popleft()

            except Exception as e:
                log.error(f"[Inference] ❌ +{h}h failed: {e}")
                log.error(traceback.format_exc())
                results[h] = {d: 3 for d in DISTRICTS}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4a — FIREBASE  (auto-retry)
# ══════════════════════════════════════════════════════════════════════════════

def firebase_set_with_retry(ref_path: str, payload: dict, max_retries: int = 3):
    """Write to Firebase with exponential back-off retry."""
    for attempt in range(1, max_retries + 1):
        try:
            firebase_db.reference(ref_path).set(payload)
            log.info(f"[Firebase] ✅ Written: {ref_path}  (attempt {attempt})")
            return True
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"[Firebase] ⚠️  Write attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                log.info(f"[Firebase]    Retrying in {wait}s …")
                time.sleep(wait)
            else:
                log.error(f"[Firebase] ❌ All {max_retries} attempts failed for {ref_path}")
                log.error(traceback.format_exc())
    return False


def push_to_firebase(firebase_ts: str, display_ts: str, pred_results: dict, source_count: int):
    """
    Write prediction results to three Firebase locations:
      GAGNN_v2/predictions/<ts>        — 6h, roadside-key format (HTML dashboard)
      GAGNN_v3/multi_horizon/<ts>      — 3h / 6h / 24h district format
      GAGNN_v3/latest                  — pointer to latest timestamp + averages
    """
    log.info(f"[Firebase] Pushing predictions ts={display_ts} (key={firebase_ts}) …")

    # v2 (已經使用安全的 V2_KEY_MAP)
    v2_payload = {V2_KEY_MAP.get(d, d): v for d, v in pred_results[6].items()}
    firebase_set_with_retry(f"GAGNN_v2/predictions/{firebase_ts}", v2_payload)

    # v3 multi-horizon — 關鍵修正：所有區域名稱都要 sanitise
    def safe_dict(pred_dict):
        return {sanitize_firebase_key(d): v for d, v in pred_dict.items()}

    v3_payload = {
        "generated_at":   display_ts,
        "model_version":  "GAGNN_v2.0",
        "source_records": source_count,
        "horizons": {
            "3h":  safe_dict(pred_results[3]),
            "6h":  safe_dict(pred_results[6]),
            "24h": safe_dict(pred_results[24]),
        },
        "averages": {
            "3h":  round(float(np.mean(list(pred_results[3].values()))), 2),
            "6h":  round(float(np.mean(list(pred_results[6].values()))), 2),
            "24h": round(float(np.mean(list(pred_results[24].values()))), 2),
        },
    }
    firebase_set_with_retry(f"GAGNN_v3/multi_horizon/{firebase_ts}", v3_payload)

    # Latest pointer
    firebase_set_with_retry("GAGNN_v3/latest", {
        "timestamp":    display_ts,
        "avg_aqhi_3h":  v3_payload["averages"]["3h"],
        "avg_aqhi_6h":  v3_payload["averages"]["6h"],
        "avg_aqhi_24h": v3_payload["averages"]["24h"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4b — DUAL CSV LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(ts_str: str, pred_results: dict):
    """
    1. Overwrite today's CSV (fast front-end retrieval)
    2. Append to history CSV (model accuracy analysis)
    """
    rows = [{
        "generated_at": ts_str,
        "district":     d,
        "aqhi_3h":      pred_results[3].get(d, -1),
        "aqhi_6h":      pred_results[6].get(d, -1),
        "aqhi_24h":     pred_results[24].get(d, -1),
    } for d in DISTRICTS]
    df = pd.DataFrame(rows)

    try:
        df.to_csv(CSV_TODAY_PATH, index=False)
        log.info(f"[CSV] ✅ Today overwritten: {CSV_TODAY_PATH}")
    except Exception as e:
        log.error(f"[CSV] ❌ Today write failed: {e}")

    try:
        write_header = not CSV_HISTORY_PATH.exists()
        df.to_csv(CSV_HISTORY_PATH, mode="a", header=write_header, index=False)
        log.info(f"[CSV] ✅ History appended:  {CSV_HISTORY_PATH}")
    except Exception as e:
        log.error(f"[CSV] ❌ History append failed: {e}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def check_drift(pred_results: dict):
    """
    Compare 6h predictions against the most recent actual AQHI from Firebase.
    If mean absolute error > 3 AQHI levels for DRIFT_WINDOW consecutive runs,
    log a retrain warning.
    """
    DRIFT_THRESHOLD = 3.0
    DRIFT_WINDOW    = 3

    try:
        ref  = firebase_db.reference("aqhi_history")
        snap = ref.order_by_key().limit_to_last(1).get()
        if not snap:
            return
        latest_key = list(snap.keys())[0]
        latest_rec = snap[latest_key]

        errors = []
        for district in DISTRICTS:
            pred_val = pred_results[6].get(district, 3)
            actual_val = None
            for k in [f"AQHI_{district}",
                      f"AQHI_{district.replace('/','_')}",
                      f"AQHI_{district.replace(' ','_')}"]:
                if k in latest_rec:
                    actual_val = float(latest_rec[k])
                    break
            if actual_val is not None:
                errors.append(abs(pred_val - actual_val))

        if not errors:
            return

        mean_err = float(np.mean(errors))
        drift_history.append(mean_err)
        if len(drift_history) > DRIFT_WINDOW:
            drift_history.popleft()

        log.info(f"[Drift] Mean |pred–actual| = {mean_err:.2f} AQHI levels  "
                 f"(vs actual record {latest_key})")

        if (len(drift_history) == DRIFT_WINDOW and
                all(e > DRIFT_THRESHOLD for e in drift_history)):
            log.warning(
                f"[Drift] 🚨 RETRAIN WARNING: Mean error exceeded {DRIFT_THRESHOLD} "
                f"AQHI levels for {DRIFT_WINDOW} consecutive predictions! "
                f"History: {list(drift_history)}. Consider retraining."
            )

    except Exception as e:
        log.warning(f"[Drift] Could not compute drift: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
device       = torch.device("cpu")
models       = {3: None, 6: None, 24: None}
feat_scaler  = None
aqhi_scaler  = None
edge_index   = None
edge_weight  = None
checksums    = {}

last_prediction_result: dict = {}
prediction_log: list         = []
sync_buffer: list            = []
latency_log  = {3: deque(), 6: deque(), 24: deque()}
drift_history: deque         = deque()

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — FIREBASE
# ══════════════════════════════════════════════════════════════════════════════
def init_firebase():
    log.info("[Firebase] Initialising Admin SDK …")
    try:
        sa_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        if sa_str:
            log.info("[Firebase] Using FIREBASE_SERVICE_ACCOUNT_JSON env var")
            cred = credentials.Certificate(json.loads(sa_str))
        else:
            local = BASE_DIR / "serviceAccountKey.json"
            if not local.exists():
                raise FileNotFoundError(
                    "Set FIREBASE_SERVICE_ACCOUNT_JSON env var "
                    "or place serviceAccountKey.json beside gnn_main.py"
                )
            log.info(f"[Firebase] Using local file {local}")
            cred = credentials.Certificate(str(local))

        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
        log.info("[Firebase] ✅ Admin SDK ready")
    except Exception as e:
        log.error(f"[Firebase] ❌ Init failed: {e}")
        log.error(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — MODELS, SCALERS, GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def _load_one_model(horizon: int) -> GAGNN | None:
    """Load GAGNN for one horizon. Tries per-horizon path, then canonical."""
    path = MODEL_PATHS[horizon]
    if not path.exists():
        if horizon == 6 and MODEL_CANONICAL.exists():
            log.warning(f"[Model] {path.name} not found — trying {MODEL_CANONICAL.name}")
            path = MODEL_CANONICAL
        else:
            log.error(f"[Model] ❌ Weight file missing: {path}")
            return None

    log.info(f"[Model] Loading +{horizon}h from {path.name} …")
    m = GAGNN(node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM,
              seq_len=SEQ_LEN, horizon=horizon).to(device)
    try:
        state = torch.load(path, map_location=device)
        m.load_state_dict(state, strict=True)
        log.info(f"[Model] ✅ +{horizon}h strict load OK")
    except RuntimeError as e:
        log.warning(f"[Model] Strict load failed for +{horizon}h: {e}")
        log.warning(f"[Model]    Attempting non-strict load …")
        state = torch.load(path, map_location=device)
        incompatible = m.load_state_dict(state, strict=False)
        log.warning(f"[Model]    Non-strict result: {incompatible}")
    m.eval()

    chk = compute_checksum(m)
    checksums[horizon] = chk
    verify_checksum(m, horizon)
    log.info(f"[Model] +{horizon}h checksum={chk}")
    return m


def init_models():
    global models, feat_scaler, aqhi_scaler, edge_index, edge_weight

    log.info("=" * 65)
    log.info("  BuildTech GAGNN v2.0 — System Initialisation")
    log.info(f"  Base dir : {BASE_DIR}")
    log.info(f"  Device   : {device}")
    log.info("=" * 65)

    # ── Scalers ────────────────────────────────────────────────────────────
    for path, name in [(FEAT_SCALER_PATH, "feat_scaler"), (AQHI_SCALER_PATH, "aqhi_scaler")]:
        log.info(f"[Startup] {name}: exists={path.exists()}  path={path}")
    try:
        feat_scaler = joblib.load(FEAT_SCALER_PATH)
        aqhi_scaler = joblib.load(AQHI_SCALER_PATH)
        log.info(f"[Startup] ✅ feat_scaler loaded  shape={feat_scaler.mean_.shape}")
        log.info(f"[Startup] ✅ aqhi_scaler loaded  "
                 f"mean={aqhi_scaler.mean_[0]:.4f}  scale={aqhi_scaler.scale_[0]:.4f}")
    except Exception as e:
        log.error(f"[Startup] ❌ Scaler load failed: {e}")
        log.error(traceback.format_exc())
        log.error("[Startup] Cannot continue without scalers.")
        return

    # ── Graph ──────────────────────────────────────────────────────────────
    try:
        edge_index, edge_weight = load_or_build_graph(device)
    except Exception as e:
        log.error(f"[Startup] ❌ Graph init failed: {e}")
        log.error(traceback.format_exc())
        return

    # ── Models ─────────────────────────────────────────────────────────────
    any_loaded = False
    for h in HORIZONS:
        m = _load_one_model(h)
        models[h] = m
        if m is not None:
            any_loaded = True

    if not any_loaded:
        log.error("[Startup] ❌ No models loaded — predictions will return fallback")
    else:
        loaded = [h for h in HORIZONS if models[h] is not None]
        log.info(f"[Startup] ✅ Models ready for horizons: {loaded}")

    log.info("[Startup] 🚀 GAGNN system initialised")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════
def fetch_aqhi_history_window() -> list:
    """Pull last SEQ_LEN records from Firebase aqhi_history, sorted oldest first."""
    log.info(f"[Fetch] Pulling last {SEQ_LEN} records from aqhi_history …")
    try:
        snap = firebase_db.reference("aqhi_history") \
                          .order_by_key().limit_to_last(SEQ_LEN).get()
        if not snap:
            log.warning("[Fetch] aqhi_history returned empty snapshot")
            return []
        records = [{"_key": k, **v} for k, v in snap.items()]
        records.sort(key=lambda r: r["_key"])
        log.info(f"[Fetch] ✅ {len(records)} records  "
                 f"{records[0]['_key']} → {records[-1]['_key']}")
        return records
    except Exception as e:
        log.error(f"[Fetch] ❌ Firebase read failed: {e}")
        log.error(traceback.format_exc())
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction_pipeline() -> dict | None:
    global last_prediction_result, prediction_log

    log.info("━" * 65)
    log.info("[Pipeline] 🔄 Starting prediction pipeline …")
    t_pipeline = time.perf_counter()

    # ── 1. Fetch ───────────────────────────────────────────────────────────
    records = fetch_aqhi_history_window()
    if sync_buffer:
        log.info(f"[Pipeline] Merging {len(sync_buffer)} records from /sync buffer")
        records = list(records) + list(sync_buffer)
        records.sort(key=lambda r: r.get("Date", r.get("_key", "")))
    if not records:
        log.error("[Pipeline] ❌ No data — aborting")
        return None

    # ── 2. Features ────────────────────────────────────────────────────────
    X_tensor, _ = build_feature_window(records)
    if X_tensor is None:
        log.error("[Pipeline] ❌ Feature build failed — aborting")
        return None

    # ── 3. Inference ───────────────────────────────────────────────────────
    pred_results = run_inference(X_tensor)

    # ── 4. Drift check ─────────────────────────────────────────────────────
    try:
        check_drift(pred_results)
    except Exception:
        pass   # non-fatal

    # ── 5. Timestamp ───────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    
    # Firebase 使用的安全 key（底線 + 秒數，避免碰撞）
    firebase_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 給人類看的漂亮時間（用在 dashboard、CSV、payload）
    display_ts = now.strftime("%Y-%m-%d %H:%M")
    
    log.info(f"[Pipeline] Timestamp: {display_ts}  (Firebase key = {firebase_ts})")

    # ── 6. Firebase ────────────────────────────────────────────────────────
    push_to_firebase(firebase_ts, display_ts, pred_results, len(records))

    # ── 7. CSV ─────────────────────────────────────────────────────────────
    save_csv(display_ts, pred_results)

    # ── 8. Cache ───────────────────────────────────────────────────────────
    avgs = {str(h)+"h": round(float(np.mean(list(pred_results[h].values()))), 2)
            for h in HORIZONS}
    last_prediction_result = {
        "generated_at":   display_ts,
        "horizons":       {"3h": pred_results[3], "6h": pred_results[6], "24h": pred_results[24]},
        "avg":            avgs,
        "source_records": len(records),
        "checksums":      checksums,
        "latency_ms": {
            str(h)+"h": round(float(np.mean(list(latency_log[h]))), 1)
                        if latency_log[h] else None
            for h in HORIZONS
        },
    }

    entry = {"timestamp": display_ts, **{f"avg_{k}": v for k, v in avgs.items()}}
    prediction_log.append(entry)
    if len(prediction_log) > 48:
        prediction_log = prediction_log[-48:]

    total_ms = (time.perf_counter() - t_pipeline) * 1000
    log.info(f"[Pipeline] ✅ Complete in {total_ms:.0f}ms  "
             f"avg AQHI → 3h:{avgs['3h']}  6h:{avgs['6h']}  24h:{avgs['24h']}")
    log.info("━" * 65)
    return last_prediction_result


# ══════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BuildTech GAGNN v2 | Prediction Service</title>
<style>
:root{--bg:#070b0f;--card:rgba(12,18,26,0.96);--blue:#38bdf8;--green:#34d399;
      --red:#f87171;--amber:#fbbf24;--text:#cbd5e1;--muted:#64748b;
      --border:rgba(56,189,248,0.13);--border2:rgba(56,189,248,0.28);}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Courier New',monospace;background:var(--bg);color:var(--text);padding:18px;min-height:100vh;}
h1{color:var(--blue);font-size:1.3rem;letter-spacing:.1em;margin-bottom:3px;}
.sub{color:var(--muted);font-size:.73rem;margin-bottom:20px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px;margin-bottom:16px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:16px;position:relative;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(56,189,248,.2),transparent);}
.card h2{font-size:.72rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;}
.big{font-size:2.6rem;font-weight:700;color:var(--blue);line-height:1;margin-bottom:5px;}
table{width:100%;border-collapse:collapse;font-size:.76rem;}
th{color:var(--muted);text-align:left;padding:4px 7px;border-bottom:1px solid var(--border);}
td{padding:4px 7px;border-bottom:1px solid rgba(56,189,248,.05);}
.low{color:var(--green);font-weight:700;} .mid{color:var(--amber);font-weight:700;} .hi{color:var(--red);font-weight:700;}
.ok{color:var(--green);} .err{color:var(--red);} .warn{color:var(--amber);}
.ep{font-size:.76rem;margin:5px 0;padding:5px 9px;background:rgba(0,0,0,.3);
    border-radius:5px;border-left:3px solid var(--blue);}
.ep a{color:var(--blue);text-decoration:none;} .ep a:hover{text-decoration:underline;}
.ts{font-size:.68rem;color:var(--muted);margin-top:7px;}
.chk{font-family:'Courier New',monospace;font-size:.68rem;color:var(--muted);}
</style>
</head>
<body>
<h1>🏗️ BuildTech GAGNN v2.0 · Prediction Service</h1>
<div class="sub">Auto-refresh 5 min &nbsp;|&nbsp; {{ server_time }}</div>

<div class="grid">
  <!-- System Status -->
  <div class="card">
    <h2>⚙️ System Status</h2>
    <div>Models (3/6/24h): <span class="{{ 'ok' if models_ok else 'err' }}">{{ '✅ All loaded' if models_ok else '⚠️ Partial / missing' }}</span></div>
    <div>Graph (.pt file): <span class="{{ 'ok' if graph_ok else 'warn' }}">{{ '✅ Persisted' if graph_ok else '⚠️ Rebuilt' }}</span></div>
    <div>Feat scaler:      <span class="{{ 'ok' if scaler_ok else 'err' }}">{{ '✅ Loaded' if scaler_ok else '❌ Missing' }}</span></div>
    <div>Firebase:         <span class="{{ 'ok' if firebase_ok else 'err' }}">{{ '✅ Connected' if firebase_ok else '❌ Not connected' }}</span></div>
    <div>Scheduler:        <span class="ok">✅ Hourly @ HH:05 HKT</span></div>
    <div class="ts">Device: {{ device }}</div>
    {% if checksums %}
    <div class="ts">Checksums:
      {% for h, c in checksums.items() %} +{{ h }}h=<span class="chk">{{ c }}</span>{% endfor %}
    </div>
    {% endif %}
    {% if drift %}
    <div class="ts warn">Drift window: {{ drift }}</div>
    {% endif %}
  </div>

  <!-- Latest Averages -->
  <div class="card">
    <h2>📈 Latest Prediction Averages</h2>
    {% if last_pred %}
    <div class="big">{{ last_pred.avg['6h'] }}</div>
    <div class="ts" style="margin-bottom:8px;">6h city-wide avg AQHI</div>
    <table>
      <tr><th>Horizon</th><th>Avg</th><th>Level</th><th>Latency</th></tr>
      {% for h_label, h_key in [('3h','3h'),('6h','6h'),('24h','24h')] %}
      <tr>
        <td>+{{ h_label }}</td>
        <td>{{ last_pred.avg[h_key] }}</td>
        <td class="{{ 'low' if last_pred.avg[h_key]<=3 else ('mid' if last_pred.avg[h_key]<=6 else 'hi') }}">
          {{ '低' if last_pred.avg[h_key]<=3 else ('中' if last_pred.avg[h_key]<=6 else '高') }}</td>
        <td class="ts">{{ last_pred.latency_ms.get(h_key,'?') }}ms</td>
      </tr>
      {% endfor %}
    </table>
    <div class="ts">Generated: {{ last_pred.generated_at }}  |  Source records: {{ last_pred.source_records }}</div>
    {% else %}
    <div class="warn">⏳ No predictions yet. POST /predict or wait for scheduler.</div>
    {% endif %}
  </div>

  <!-- Endpoints -->
  <div class="card">
    <h2>🔌 API Endpoints</h2>
    <div class="ep">GET <a href="/status">/status</a> — JSON health</div>
    <div class="ep">GET <a href="/predictions">/predictions</a> — Latest prediction JSON</div>
    <div class="ep">GET <a href="/predictions/csv">/predictions/csv</a> — CSV download</div>
    <div class="ep">GET <a href="/history">/history</a> — Last 48 runs</div>
    <div class="ep">GET <a href="/history/csv">/history/csv</a> — History CSV</div>
    <div class="ep">GET <a href="/metrics">/metrics</a> — Latency &amp; drift</div>
    <div class="ep">GET <a href="/firebase/test">/firebase/test</a> — Ping Firebase</div>
    <div class="ep">POST /predict — Trigger now</div>
    <div class="ep">POST /sync — Receive data (GitHub Action)</div>
  </div>
</div>

<!-- Per-district table -->
{% if last_pred %}
<div class="card">
  <h2>🗺️ Per-District Forecast (18 Districts)</h2>
  <table>
    <tr><th>District</th><th>+3h AQHI</th><th>+6h AQHI</th><th>+24h AQHI</th></tr>
    {% for d in districts %}
    {% set v3 = last_pred.horizons['3h'].get(d,0) %}
    {% set v6 = last_pred.horizons['6h'].get(d,0) %}
    {% set v24 = last_pred.horizons['24h'].get(d,0) %}
    <tr>
      <td>{{ d }}</td>
      <td class="{{ 'low' if v3<=3 else ('mid' if v3<=6 else 'hi') }}">{{ v3 }}</td>
      <td class="{{ 'low' if v6<=3 else ('mid' if v6<=6 else 'hi') }}">{{ v6 }}</td>
      <td class="{{ 'low' if v24<=3 else ('mid' if v24<=6 else 'hi') }}">{{ v24 }}</td>
    </tr>
    {% endfor %}
  </table>
  <div class="ts">■ <span class="low">Low (1-3)</span> &nbsp; ■ <span class="mid">Mid (4-6)</span> &nbsp; ■ <span class="hi">High (7+)</span></div>
</div>
{% endif %}

<!-- History log -->
{% if pred_log %}
<div class="card" style="margin-top:12px;">
  <h2>📜 History Log (latest {{ pred_log|length }} pipeline runs)</h2>
  <table>
    <tr><th>Timestamp</th><th>Avg +3h</th><th>Avg +6h</th><th>Avg +24h</th></tr>
    {% for entry in pred_log|reverse %}
    <tr>
      <td>{{ entry.timestamp }}</td>
      <td>{{ entry.get('avg_3h','--') }}</td>
      <td>{{ entry.get('avg_6h','--') }}</td>
      <td>{{ entry.get('avg_24h','--') }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}
</body></html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def dashboard():
    log.info("[Route] GET /")
    return render_template_string(
        DASHBOARD_HTML,
        server_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        models_ok   = all(models[h] is not None for h in HORIZONS),
        graph_ok    = EDGE_INDEX_PATH.exists(),
        scaler_ok   = feat_scaler is not None and aqhi_scaler is not None,
        firebase_ok = len(firebase_admin._apps) > 0,
        device      = str(device),
        checksums   = {f"{h}h": v for h, v in checksums.items()},
        drift       = list(drift_history) if drift_history else None,
        last_pred   = last_prediction_result or None,
        pred_log    = prediction_log,
        districts   = DISTRICTS,
    )


@app.route("/status", methods=["GET"])
def status():
    log.info("[Route] GET /status")
    return jsonify({
        "status":         "healthy" if all(models[h] is not None for h in HORIZONS) else "degraded",
        "version":        "GAGNN_v2.0",
        "models_loaded":  {f"{h}h": models[h] is not None for h in HORIZONS},
        "scalers_loaded": feat_scaler is not None and aqhi_scaler is not None,
        "graph_persisted": EDGE_INDEX_PATH.exists(),
        "firebase_ok":    len(firebase_admin._apps) > 0,
        "checksums":      {f"{h}h": checksums.get(h) for h in HORIZONS},
        "device":         str(device),
        "last_pred_at":   last_prediction_result.get("generated_at", "never"),
        "horizons":       HORIZONS,
        "server_time":    datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route("/sync", methods=["POST"])
def sync_data():
    log.info("[Route] POST /sync")
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Empty body"}), 400

        date_val  = data.get("Date", "unknown")
        aqhi_keys = [k for k in data if k.startswith("AQHI_")]
        log.info(f"[Sync] Received Date={date_val}  keys={len(data)}  AQHI_keys={len(aqhi_keys)}")

        # Label sanitiser: log any AQHI keys that don't match DISTRICTS
        unmatched = []
        for k in aqhi_keys:
            if sanitise_label(k) is None:
                unmatched.append(k)
        if unmatched:
            log.warning(f"[Sync] Unmatched AQHI keys (will use fallback AQHI=3): {unmatched}")

        sync_buffer.append(data)
        if len(sync_buffer) > SEQ_LEN * 2:
            sync_buffer.pop(0)

        log.info(f"[Sync] ✅ Buffer size now: {len(sync_buffer)}")
        return jsonify({
            "status":        "success",
            "received_date": date_val,
            "buffer_size":   len(sync_buffer),
            "aqhi_keys_ok":  len(aqhi_keys) - len(unmatched),
            "aqhi_keys_bad": len(unmatched),
        }), 200
    except Exception as e:
        log.error(f"[Sync] ❌ {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_now():
    log.info("[Route] POST /predict — manual trigger")
    try:
        result = run_prediction_pipeline()
        if result:
            return jsonify({"status": "success", "result": result}), 200
        return jsonify({"status": "error", "message": "Pipeline failed — check Render logs"}), 500
    except Exception as e:
        log.error(f"[Route /predict] ❌ {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/predictions", methods=["GET"])
def get_predictions():
    log.info("[Route] GET /predictions")
    if last_prediction_result:
        return jsonify({"status": "success", "data": last_prediction_result}), 200

    # CSV fallback
    if CSV_TODAY_PATH.exists():
        try:
            df = pd.read_csv(CSV_TODAY_PATH)
            log.info("[Route] Serving CSV fallback")
            return jsonify({"status": "csv_fallback", "data": df.to_dict("records")}), 200
        except Exception as e:
            log.error(f"[Route] CSV read error: {e}")

    return jsonify({"error": "No predictions yet. POST /predict to run pipeline."}), 404


@app.route("/predictions/csv", methods=["GET"])
def get_predictions_csv():
    log.info("[Route] GET /predictions/csv")
    if not CSV_TODAY_PATH.exists():
        return jsonify({"error": "CSV not found. Run /predict first."}), 404
    try:
        return Response(
            CSV_TODAY_PATH.read_text(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gagnn_prediction_today.csv"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    log.info("[Route] GET /history")
    limit = request.args.get("limit", 48, type=int)
    return jsonify({"status": "success", "count": len(prediction_log),
                    "data": prediction_log[-limit:]}), 200


@app.route("/history/csv", methods=["GET"])
def get_history_csv():
    log.info("[Route] GET /history/csv")
    if not CSV_HISTORY_PATH.exists():
        return jsonify({"error": "History CSV not found yet."}), 404
    try:
        return Response(
            CSV_HISTORY_PATH.read_text(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gagnn_prediction_history.csv"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Inference latency and drift report."""
    log.info("[Route] GET /metrics")
    lat = {}
    for h in HORIZONS:
        vals = list(latency_log[h])
        lat[f"{h}h"] = {
            "samples": len(vals),
            "mean_ms": round(float(np.mean(vals)), 1) if vals else None,
            "min_ms":  round(float(np.min(vals)),  1) if vals else None,
            "max_ms":  round(float(np.max(vals)),  1) if vals else None,
        }
    return jsonify({
        "latency":       lat,
        "drift_history": list(drift_history),
        "drift_warning": (
            len(drift_history) >= 3 and all(e > 3.0 for e in list(drift_history)[-3:])
        ),
    }), 200


@app.route("/firebase/test", methods=["GET"])
def firebase_test():
    log.info("[Route] GET /firebase/test")
    try:
        ts  = datetime.now(timezone.utc).isoformat()
        ref = "GAGNN_v3/health_check"
        firebase_db.reference(ref).set({"status": "ok", "tested_at": ts})
        val = firebase_db.reference(ref).get()
        log.info(f"[Firebase Test] ✅ Echo: {val}")
        return jsonify({"status": "success", "echo": val}), 200
    except Exception as e:
        log.error(f"[Firebase Test] ❌ {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
def start_scheduler():
    sched = BackgroundScheduler(timezone="Asia/Hong_Kong")
    sched.add_job(run_prediction_pipeline, "cron", minute=5,
                  id="hourly", replace_existing=True)
    sched.add_job(run_prediction_pipeline, "date",
                  run_date=datetime.now() + timedelta(minutes=1),
                  id="startup_run")
    sched.start()
    log.info("[Scheduler] ✅ Started  — hourly at HH:05 HKT + startup run in 1 min")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTOR (v2.0 Aligned)
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="GAGNN v2.0 Training Script")
    p.add_argument("--csv",        type=str, default="./data/", help="Path to CSV data directory")
    p.add_argument("--out-dir",    type=str, default="./",      help="Output directory for weights/scalers")
    p.add_argument("--epochs",     type=int, default=100,       help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=32,        help="Batch size")
    p.add_argument("--lr",         type=float, default=0.001,   help="Learning rate")
    p.add_argument("--horizons",   type=int, nargs="+",         help="List of horizons to train (e.g. 3 6 24)")
    p.add_argument("--plot",       action="store_true",         help="Save loss curves as PNG")
    p.add_argument("--eval-only",  action="store_true",         help="Load saved weights and evaluate only")
    p.add_argument("--horizon",    type=int, default=6,         help="Single horizon for --eval-only mode")
    return p.parse_args()

def train_all_horizons(args):
    """迴圈執行 3h, 6h, 24h 的訓練"""
    for h in args.horizons:
        print("\n" + "="*50)
        print(f"🚩 STARTING TRAINING FOR HORIZON: {h}h")
        print("="*50)
        # 對齊你檔案中的函數名稱: main_train_flow
        main_train_flow(h, args) 

if __name__ == "__main__":
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.horizons is None:
        args.horizons = [3, 6, 24]

    if args.eval_only:
        # 如果只想測試單一模型
        main_train_flow(args.horizon, args)
    else:
        print(f"🚀 [Log] Starting GAGNN v2.0 training: {args.horizons}")
        train_all_horizons(args)
    
    print("\n✅ [Log] Training script finished successfully.")
    # 這裡絕不放 app.run()，確保 GitHub Action 執行完會自動關閉
