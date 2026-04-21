"""
gnn_main.py — BuildTech GAGNN Prediction Service
=================================================
Runs on Render (https://buildtech-gnn-service.onrender.com/)

Endpoints
  GET  /                → health check + last prediction summary (HTML dashboard)
  GET  /status          → JSON health + model info
  POST /sync            → receive raw hourly data from GitHub Action
  GET  /predictions     → latest prediction JSON
  GET  /predictions/csv → download prediction CSV
  POST /predict         → trigger inference manually (optional)
  GET  /history         → last N prediction records as JSON

Pipeline (auto-runs every hour via APScheduler)
  1. Pull latest record from Firebase aqhi_history
  2. Assemble 24-hour feature window
  3. Run HK_Pro_Model → forecast horizons [3h, 6h, 24h]
  4. Post-process (clip 1-11, round)
  5. Push results to Firebase: GAGNN_v2/predictions/<timestamp>
  6. Write gagnn_prediction_today.csv
  7. Update /predictions endpoint cache

Requirements (add to requirements.txt on Render):
  flask
  torch
  torch_geometric
  joblib
  pandas
  numpy
  scipy
  firebase-admin
  apscheduler
  requests
  gunicorn
"""

import os
import sys
import math
import time
import logging
import traceback
import json
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ── Firebase Admin SDK ────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ── Flask ─────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, Response, render_template_string

# ── Scheduler ─────────────────────────────────────────────────────────────
from apscheduler.schedulers.background import BackgroundScheduler

# ── PyG ───────────────────────────────────────────────────────────────────
from torch_geometric.nn import GATConv

# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("GAGNN")

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS & PATHS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR         = Path(__file__).parent
MODEL_PATH       = BASE_DIR / "hk_pro_model_best.pth"
FEAT_SCALER_PATH = BASE_DIR / "feat_scaler.pkl"
AQHI_SCALER_PATH = BASE_DIR / "aqhi_scaler.pkl"
CSV_OUTPUT_PATH  = BASE_DIR / "gagnn_prediction_today.csv"
HISTORY_CSV_PATH = BASE_DIR / "gagnn_prediction_history.csv"

# Model hyper-params — must match training config
NODE_FEATURES = 10
HIDDEN_DIM    = 128
SEQ_LEN       = 24
HORIZONS      = [3, 6, 24]   # predict 3h, 6h and 24h ahead
N_STATIONS    = 18

FIREBASE_DB_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app"

# ══════════════════════════════════════════════════════════════════════════
# STATION DEFINITIONS (must match train_model.py exactly)
# ══════════════════════════════════════════════════════════════════════════
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
    frozenset({"Kwai Chung", "Central/Western"}):  0.40,
    frozenset({"Tuen Mun",   "Tsuen Wan"}):        0.50,
    frozenset({"Tung Chung", "Tsuen Wan"}):        0.45,
    frozenset({"Tap Mun",    "Sha Tin"}):          0.40,
    frozenset({"North",      "Tai Po"}):           0.60,
}

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

# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (mirrors model_pro.py / train_model.py)
# ══════════════════════════════════════════════════════════════════════════
class HK_Pro_Model(nn.Module):
    """GAT-based spatio-temporal model. Must stay identical to training."""
    def __init__(self, node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 seq_len=SEQ_LEN, horizon=6):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon

        self.temporal = nn.GRU(
            input_size=node_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.fc_out = nn.Linear(hidden_dim, horizon)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        """x: [B, N, T, F]"""
        B, N, T, F = x.shape
        device = x.device

        edge_index_all = []
        for i in range(B):
            edge_index_all.append(edge_index + i * N)
        batched_edge_index = torch.cat(edge_index_all, dim=1).to(device)

        x = x.reshape(B * N, T, F)
        temporal_out, _ = self.temporal(x)
        temporal_out = temporal_out[:, -1, :]

        spatial_out = self.gat1(temporal_out, batched_edge_index)
        spatial_out = self.relu(spatial_out)
        spatial_out = self.dropout(spatial_out)

        spatial_out = self.gat2(spatial_out, batched_edge_index)
        spatial_out = self.relu(spatial_out)

        out = self.fc_out(spatial_out)
        out = out.view(B, N, self.horizon)
        return out

# ══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def build_graph(max_dist_km=40.0):
    n = len(DISTRICTS)
    src, dst = [], []
    for i in range(n):
        lat1, lon1 = DISTRICT_COORDS[DISTRICTS[i]]
        for j in range(n):
            if i == j:
                continue
            lat2, lon2 = DISTRICT_COORDS[DISTRICTS[j]]
            d = _haversine_km(lat1, lon1, lat2, lon2)
            if d <= max_dist_km:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    log.info(f"[Graph] Built: {n} nodes, {edge_index.shape[1]} directed edges")
    return edge_index

# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (mirrors train_model.py)
# ══════════════════════════════════════════════════════════════════════════
def _stn_mean(record: dict, prefix: str, stations: list) -> float:
    vals = [record.get(f"{prefix}{s}") for s in stations]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else 0.0

def _circular_mean_deg(record: dict, prefix: str, stations: list) -> float:
    dirs = [record.get(f"{prefix}{s}") for s in stations]
    dirs = [d for d in dirs if d is not None]
    if not dirs:
        return 0.0
    sin_m = np.mean([math.sin(math.radians(d)) for d in dirs])
    cos_m = np.mean([math.cos(math.radians(d)) for d in dirs])
    deg   = math.degrees(math.atan2(sin_m, cos_m))
    return (deg % 360 + 360) % 360

def record_to_node_features(record: dict, timestamp: datetime) -> np.ndarray:
    """
    Convert one aqhi_history record to [N, F] node feature array.
    Features (10): aqhi, humidity, wspd, wdir_sin, wdir_cos,
                   cyclone, hour_sin, hour_cos, dow_sin, dow_cos
    """
    h_sin = math.sin(2 * math.pi * timestamp.hour / 24)
    h_cos = math.cos(2 * math.pi * timestamp.hour / 24)
    d_sin = math.sin(2 * math.pi * timestamp.weekday() / 7)
    d_cos = math.cos(2 * math.pi * timestamp.weekday() / 7)
    cyclone = float(record.get("Cyclone_Present", 0) or 0)

    rows = []
    for district in DISTRICTS:
        # AQHI — try multiple key variants
        aqhi = None
        for k in [f"AQHI_{district}",
                  f"AQHI_{district.replace('/', '_').replace(' ','_')}",
                  f"AQHI_{district.replace(' ','_')}"]:
            if k in record:
                aqhi = float(record[k])
                break
        aqhi = float(np.clip(aqhi if aqhi is not None else 3, 1, 11))

        hum   = _stn_mean(record, "HUM_",  DISTRICT_HUM_MAP[district])
        hum   = float(np.clip(hum, 0, 100))
        wspd  = _stn_mean(record, "WSPD_", DISTRICT_WIND_MAP[district])
        wdir  = _circular_mean_deg(record, "PDIR_", DISTRICT_WIND_MAP[district])
        wdir_rad = math.radians(wdir)

        rows.append([
            aqhi,
            hum,
            wspd,
            math.sin(wdir_rad),
            math.cos(wdir_rad),
            cyclone,
            h_sin, h_cos,
            d_sin, d_cos,
        ])

    return np.array(rows, dtype=np.float32)   # [N, 10]

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════
model_3h  = None   # horizon=3
model_6h  = None   # horizon=6
model_24h = None   # horizon=24
feat_scaler  = None
aqhi_scaler  = None
edge_index   = None
device       = torch.device("cpu")

last_prediction_result = {}   # cached for /predictions endpoint
prediction_log         = []   # rolling log for /history endpoint
sync_buffer            = []   # hourly records received via /sync

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════
# STARTUP: FIREBASE + MODEL LOAD
# ══════════════════════════════════════════════════════════════════════════
def init_firebase():
    """
    Initialise Firebase Admin SDK.
    On Render, set the env var FIREBASE_SERVICE_ACCOUNT_JSON to the full
    service-account JSON string (Settings → Environment → Add).
    Falls back to a local serviceAccountKey.json file for local dev.
    """
    log.info("[Firebase] Initialising Admin SDK …")

    try:
        sa_json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        if sa_json_str:
            log.info("[Firebase] Using env-var service account")
            sa_info = json.loads(sa_json_str)
            cred = credentials.Certificate(sa_info)
        else:
            local_key = BASE_DIR / "serviceAccountKey.json"
            if local_key.exists():
                log.info(f"[Firebase] Using local file: {local_key}")
                cred = credentials.Certificate(str(local_key))
            else:
                raise FileNotFoundError(
                    "No Firebase credentials found. "
                    "Set FIREBASE_SERVICE_ACCOUNT_JSON env var or place serviceAccountKey.json beside gnn_main.py"
                )

        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
        log.info("[Firebase] ✅ Admin SDK ready")

    except Exception as e:
        log.error(f"[Firebase] ❌ Init failed: {e}")
        log.error(traceback.format_exc())
        # App continues; predictions will still work but Firebase writes will fail


def load_model_for_horizon(horizon: int) -> HK_Pro_Model:
    """Load one model instance for a given output horizon."""
    log.info(f"[Model] Loading weights for horizon={horizon}h …")
    m = HK_Pro_Model(
        node_features=NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        seq_len=SEQ_LEN,
        horizon=horizon,
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    # If weights were saved for a different horizon we attempt partial load
    try:
        m.load_state_dict(state, strict=True)
        log.info(f"[Model] ✅ Loaded horizon={horizon}h (strict)")
    except RuntimeError as e:
        log.warning(f"[Model] Strict load failed for horizon={horizon}h: {e}")
        # Allow non-strict load; fc_out will be re-initialised randomly
        incompatible = m.load_state_dict(state, strict=False)
        log.warning(f"[Model] Non-strict load: {incompatible}")
    m.eval()
    return m


def init_models():
    global model_3h, model_6h, model_24h, feat_scaler, aqhi_scaler, edge_index

    log.info("=" * 60)
    log.info("[Startup] Loading GAGNN models and scalers …")
    log.info(f"[Startup] BASE_DIR : {BASE_DIR}")
    log.info(f"[Startup] MODEL    : {MODEL_PATH}  exists={MODEL_PATH.exists()}")
    log.info(f"[Startup] FEAT_SCL : {FEAT_SCALER_PATH}  exists={FEAT_SCALER_PATH.exists()}")
    log.info(f"[Startup] AQHI_SCL : {AQHI_SCALER_PATH}  exists={AQHI_SCALER_PATH.exists()}")
    log.info(f"[Startup] Device   : {device}")

    missing = [p for p in [MODEL_PATH, FEAT_SCALER_PATH, AQHI_SCALER_PATH] if not p.exists()]
    if missing:
        log.error(f"[Startup] ❌ Missing files: {missing}")
        log.error("[Startup] Predictions will return fallback values until files are present.")
        return

    try:
        feat_scaler = joblib.load(FEAT_SCALER_PATH)
        aqhi_scaler = joblib.load(AQHI_SCALER_PATH)
        log.info("[Startup] ✅ Scalers loaded")
        log.info(f"[Startup]   feat_scaler: mean shape {feat_scaler.mean_.shape}")
        log.info(f"[Startup]   aqhi_scaler: mean={aqhi_scaler.mean_[0]:.4f} scale={aqhi_scaler.scale_[0]:.4f}")
    except Exception as e:
        log.error(f"[Startup] ❌ Scaler load error: {e}")
        log.error(traceback.format_exc())
        return

    try:
        model_3h  = load_model_for_horizon(3)
        model_6h  = load_model_for_horizon(6)
        model_24h = load_model_for_horizon(24)
        log.info("[Startup] ✅ All 3 model horizons loaded")
    except Exception as e:
        log.error(f"[Startup] ❌ Model load error: {e}")
        log.error(traceback.format_exc())
        return

    try:
        edge_index = build_graph()
        log.info(f"[Startup] ✅ Graph built: edge_index shape {edge_index.shape}")
    except Exception as e:
        log.error(f"[Startup] ❌ Graph build error: {e}")
        log.error(traceback.format_exc())

    log.info("[Startup] 🚀 GAGNN system ready")
    log.info("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════
def fetch_aqhi_history_window() -> list[dict]:
    """
    Pull the last SEQ_LEN (24) hourly records from Firebase aqhi_history.
    Returns list of records sorted oldest → newest.
    """
    log.info(f"[Fetch] Pulling last {SEQ_LEN} records from Firebase aqhi_history …")
    try:
        ref  = firebase_db.reference("aqhi_history")
        snap = ref.order_by_key().limit_to_last(SEQ_LEN).get()
        if not snap:
            log.warning("[Fetch] aqhi_history returned empty snapshot")
            return []
        records = [{"_key": k, **v} for k, v in snap.items()]
        records.sort(key=lambda r: r["_key"])
        log.info(f"[Fetch] ✅ Got {len(records)} records, "
                 f"range: {records[0]['_key']} → {records[-1]['_key']}")
        return records
    except Exception as e:
        log.error(f"[Fetch] ❌ Firebase fetch failed: {e}")
        log.error(traceback.format_exc())
        return []


def build_feature_window(records: list[dict]) -> np.ndarray | None:
    """
    Convert a list of hourly records into a normalised [1, N, SEQ_LEN, F] tensor.
    Returns None if insufficient data.
    """
    if len(records) < SEQ_LEN:
        log.warning(f"[Feature] Only {len(records)} records, need {SEQ_LEN}. "
                    f"Padding with copies of earliest record.")
        # Pad at the front
        pad = [records[0]] * (SEQ_LEN - len(records))
        records = pad + records

    log.info(f"[Feature] Building feature window from {len(records)} records …")

    frames = []
    for rec in records[-SEQ_LEN:]:
        date_str = rec.get("Date") or rec.get("_key", "").replace("_", " ").replace("-", ":")
        try:
            # Parse "2026-04-21 12:00" or "2026-04-21_12-00"
            date_str_clean = rec.get("Date") or rec["_key"].replace("_", " ", 1).replace("-", ":", 2)
            ts = datetime.strptime(date_str_clean[:16], "%Y-%m-%d %H:%M")
        except Exception:
            ts = datetime.now(timezone.utc)
            log.warning(f"[Feature] Could not parse date '{date_str}', using now()")

        node_feats = record_to_node_features(rec, ts)  # [N, F]
        frames.append(node_feats)

    X = np.stack(frames, axis=1)  # [N, SEQ_LEN, F]

    # Normalise using feat_scaler
    N, T, F = X.shape
    X_flat   = X.reshape(-1, F)
    X_norm   = feat_scaler.transform(X_flat).reshape(N, T, F)

    # Add batch dim → [1, N, SEQ_LEN, F]
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)

    log.info(f"[Feature] ✅ Feature tensor shape: {X_tensor.shape}")
    return X_tensor


def run_inference(X_tensor: torch.Tensor) -> dict:
    """
    Run all 3 horizon models on the feature tensor.
    Returns dict of {horizon_h: {district: aqhi_value, ...}, ...}
    """
    if any(m is None for m in [model_3h, model_6h, model_24h]):
        log.error("[Inference] ❌ One or more models not loaded — returning fallback")
        fallback = {d: 3 for d in DISTRICTS}
        return {3: fallback, 6: fallback, 24: fallback}

    results = {}
    X_tensor = X_tensor.to(device)
    edge_idx  = edge_index.to(device)

    model_map = {3: model_3h, 6: model_6h, 24: model_24h}

    with torch.no_grad():
        for h, mdl in model_map.items():
            log.info(f"[Inference] Running horizon={h}h …")
            try:
                t0   = time.time()
                pred = mdl(X_tensor, edge_idx)  # [1, N, horizon]
                # Take the last step of the horizon output
                pred_last = pred[0, :, -1].cpu().numpy()  # [N]

                # Denormalise
                pred_aqhi = pred_last * aqhi_scaler.scale_[0] + aqhi_scaler.mean_[0]
                pred_aqhi = np.clip(np.round(pred_aqhi), 1, 11).astype(int)

                district_preds = {DISTRICTS[i]: int(pred_aqhi[i]) for i in range(N_STATIONS)}
                results[h] = district_preds

                elapsed = (time.time() - t0) * 1000
                avg_aqhi = float(np.mean(pred_aqhi))
                log.info(f"[Inference] ✅ horizon={h}h  avg_AQHI={avg_aqhi:.2f}  "
                         f"min={pred_aqhi.min()} max={pred_aqhi.max()}  ({elapsed:.1f}ms)")
                log.info(f"[Inference]    Per-district: {district_preds}")

            except Exception as e:
                log.error(f"[Inference] ❌ horizon={h}h failed: {e}")
                log.error(traceback.format_exc())
                results[h] = {d: 3 for d in DISTRICTS}

    return results


def push_to_firebase(timestamp_str: str, pred_results: dict, source_records: list):
    """
    Write prediction results to Firebase at:
      GAGNN_v2/predictions/<timestamp>   → {district: aqhi, ...}  (6h only for dashboard compat)
      GAGNN_v3/multi_horizon/<timestamp> → {3: {...}, 6: {...}, 24: {...}}
    """
    log.info(f"[Firebase] Writing predictions for timestamp={timestamp_str} …")

    try:
        # ── GAGNN_v2/predictions (6h, dashboard backward-compat) ──────────
        # Keys use station roadside names expected by existing dashboard
        key_map = {
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
        v2_payload = {key_map.get(d, d): v for d, v in pred_results[6].items()}
        firebase_db.reference(f"GAGNN_v2/predictions/{timestamp_str}").set(v2_payload)
        log.info(f"[Firebase] ✅ GAGNN_v2/predictions/{timestamp_str} written")

        # ── GAGNN_v3/multi_horizon (3h / 6h / 24h) ────────────────────────
        v3_payload = {
            "generated_at": timestamp_str,
            "horizons": {
                "3h":  pred_results[3],
                "6h":  pred_results[6],
                "24h": pred_results[24],
            },
            "source_count": len(source_records),
            "model_version": "HK_Pro_GAGNN_v3",
        }
        firebase_db.reference(f"GAGNN_v3/multi_horizon/{timestamp_str}").set(v3_payload)
        log.info(f"[Firebase] ✅ GAGNN_v3/multi_horizon/{timestamp_str} written")

        # ── Summary/latest pointer ─────────────────────────────────────────
        firebase_db.reference("GAGNN_v3/latest").set({
            "timestamp": timestamp_str,
            "avg_aqhi_3h":  float(np.mean(list(pred_results[3].values()))),
            "avg_aqhi_6h":  float(np.mean(list(pred_results[6].values()))),
            "avg_aqhi_24h": float(np.mean(list(pred_results[24].values()))),
        })
        log.info("[Firebase] ✅ GAGNN_v3/latest updated")

    except Exception as e:
        log.error(f"[Firebase] ❌ Write failed: {e}")
        log.error(traceback.format_exc())


def save_csv(timestamp_str: str, pred_results: dict):
    """Write prediction CSVs."""
    log.info(f"[CSV] Writing prediction CSVs …")
    try:
        rows = []
        for district in DISTRICTS:
            rows.append({
                "generated_at": timestamp_str,
                "district":     district,
                "aqhi_3h":      pred_results[3].get(district, "--"),
                "aqhi_6h":      pred_results[6].get(district, "--"),
                "aqhi_24h":     pred_results[24].get(district, "--"),
            })
        df = pd.DataFrame(rows)

        # Today's prediction (overwrite)
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        log.info(f"[CSV] ✅ Today's CSV saved: {CSV_OUTPUT_PATH}")

        # History (append)
        write_header = not HISTORY_CSV_PATH.exists()
        df.to_csv(HISTORY_CSV_PATH, mode='a', header=write_header, index=False)
        log.info(f"[CSV] ✅ History CSV appended: {HISTORY_CSV_PATH}")

        return df
    except Exception as e:
        log.error(f"[CSV] ❌ Save failed: {e}")
        log.error(traceback.format_exc())
        return None


def run_prediction_pipeline():
    """
    Full pipeline:
    Fetch Firebase → Build features → Inference → Push Firebase → Save CSV → Update cache
    Called by scheduler every hour and by /predict endpoint.
    """
    global last_prediction_result, prediction_log

    log.info("━" * 60)
    log.info("[Pipeline] 🔄 Starting prediction pipeline …")
    t_start = time.time()

    # ── 1. Fetch data ────────────────────────────────────────────────────
    records = fetch_aqhi_history_window()

    # Also include any records received via /sync that aren't in Firebase yet
    if sync_buffer:
        log.info(f"[Pipeline] Merging {len(sync_buffer)} records from /sync buffer")
        records = records + sync_buffer
        records.sort(key=lambda r: r.get("Date", r.get("_key", "")))

    if not records:
        log.error("[Pipeline] ❌ No data available — aborting pipeline")
        return None

    # ── 2. Build features ────────────────────────────────────────────────
    X_tensor = build_feature_window(records)
    if X_tensor is None:
        log.error("[Pipeline] ❌ Feature build failed — aborting")
        return None

    # ── 3. Inference ─────────────────────────────────────────────────────
    pred_results = run_inference(X_tensor)

    # ── 4. Timestamp ─────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y-%m-%d %H:%M")
    log.info(f"[Pipeline] Prediction timestamp: {ts_str}")

    # ── 5. Push to Firebase ──────────────────────────────────────────────
    push_to_firebase(ts_str, pred_results, records)

    # ── 6. Save CSV ──────────────────────────────────────────────────────
    df = save_csv(ts_str, pred_results)

    # ── 7. Update in-memory cache ─────────────────────────────────────────
    last_prediction_result = {
        "generated_at": ts_str,
        "horizons": {
            "3h":  pred_results[3],
            "6h":  pred_results[6],
            "24h": pred_results[24],
        },
        "avg": {
            "3h":  round(float(np.mean(list(pred_results[3].values()))), 2),
            "6h":  round(float(np.mean(list(pred_results[6].values()))), 2),
            "24h": round(float(np.mean(list(pred_results[24].values()))), 2),
        },
        "source_records": len(records),
        "model_files": {
            "weights":      str(MODEL_PATH),
            "feat_scaler":  str(FEAT_SCALER_PATH),
            "aqhi_scaler":  str(AQHI_SCALER_PATH),
        }
    }

    log_entry = {
        "timestamp": ts_str,
        "avg_3h":  last_prediction_result["avg"]["3h"],
        "avg_6h":  last_prediction_result["avg"]["6h"],
        "avg_24h": last_prediction_result["avg"]["24h"],
    }
    prediction_log.append(log_entry)
    if len(prediction_log) > 48:   # keep last 48 entries (2 days)
        prediction_log = prediction_log[-48:]

    elapsed = time.time() - t_start
    log.info(f"[Pipeline] ✅ Pipeline complete in {elapsed:.2f}s")
    log.info(f"[Pipeline]   Avg AQHI → 3h: {last_prediction_result['avg']['3h']}  "
             f"6h: {last_prediction_result['avg']['6h']}  "
             f"24h: {last_prediction_result['avg']['24h']}")
    log.info("━" * 60)

    return last_prediction_result

# ══════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD TEMPLATE
# ══════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BuildTech GAGNN | Prediction Service</title>
<style>
  :root{--bg:#080c10;--card:rgba(14,20,28,0.95);--blue:#38bdf8;--green:#34d399;--red:#f87171;--amber:#fbbf24;--text:#cbd5e1;--muted:#64748b;--border:rgba(56,189,248,0.15);}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Courier New',monospace;background:var(--bg);color:var(--text);padding:20px;min-height:100vh;}
  h1{color:var(--blue);font-size:1.4rem;letter-spacing:0.1em;margin-bottom:4px;}
  .sub{color:var(--muted);font-size:0.75rem;margin-bottom:24px;}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:20px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;}
  .card h2{font-size:0.75rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;}
  .big{font-size:2.8rem;font-weight:700;color:var(--blue);line-height:1;margin-bottom:6px;}
  table{width:100%;border-collapse:collapse;font-size:0.78rem;}
  th{color:var(--muted);text-align:left;padding:4px 8px;border-bottom:1px solid var(--border);}
  td{padding:5px 8px;border-bottom:1px solid rgba(56,189,248,0.06);}
  .low{color:var(--green);}  .mid{color:var(--amber);}  .hi{color:var(--red);}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700;}
  .badge-blue{background:rgba(56,189,248,0.12);color:var(--blue);border:1px solid var(--border);}
  .status-ok{color:var(--green);} .status-err{color:var(--red);}
  .endpoint{font-size:0.78rem;margin:6px 0;padding:6px 10px;background:rgba(0,0,0,0.3);border-radius:6px;border-left:3px solid var(--blue);}
  .endpoint a{color:var(--blue);text-decoration:none;}
  .endpoint a:hover{text-decoration:underline;}
  .ts{font-size:0.7rem;color:var(--muted);margin-top:8px;}
  .warn{color:var(--amber);}
</style>
</head>
<body>
<h1>🏗️ BuildTech GAGNN · Prediction Service</h1>
<div class="sub">Auto-refreshes every 5 min &nbsp;|&nbsp; {{ server_time }}</div>

<div class="grid">
  <!-- Model Status -->
  <div class="card">
    <h2>⚙️ System Status</h2>
    <div style="margin-bottom:12px;">
      <div>Model weights: <span class="{{ 'status-ok' if model_ok else 'status-err' }}">{{ '✅ Loaded' if model_ok else '❌ Missing' }}</span></div>
      <div>Feat scaler:   <span class="{{ 'status-ok' if scaler_ok else 'status-err' }}">{{ '✅ Loaded' if scaler_ok else '❌ Missing' }}</span></div>
      <div>Firebase:      <span class="{{ 'status-ok' if firebase_ok else 'status-err' }}">{{ '✅ Connected' if firebase_ok else '❌ Not connected' }}</span></div>
      <div>Scheduler:     <span class="status-ok">✅ Running (hourly)</span></div>
    </div>
    <div class="ts">Device: {{ device }}</div>
  </div>

  <!-- Latest Prediction Averages -->
  <div class="card">
    <h2>📈 Latest Prediction Averages</h2>
    {% if last_pred %}
    <div class="big">{{ last_pred.avg['6h'] }}</div>
    <div style="font-size:0.8rem;color:var(--muted);margin-bottom:10px;">6h avg AQHI</div>
    <table>
      <tr><th>Horizon</th><th>Avg AQHI</th><th>Level</th></tr>
      <tr><td>+3h</td><td>{{ last_pred.avg['3h'] }}</td><td class="{{ 'low' if last_pred.avg['3h']<=3 else ('mid' if last_pred.avg['3h']<=6 else 'hi') }}">{{ '低' if last_pred.avg['3h']<=3 else ('中' if last_pred.avg['3h']<=6 else '高') }}</td></tr>
      <tr><td>+6h</td><td>{{ last_pred.avg['6h'] }}</td><td class="{{ 'low' if last_pred.avg['6h']<=3 else ('mid' if last_pred.avg['6h']<=6 else 'hi') }}">{{ '低' if last_pred.avg['6h']<=3 else ('中' if last_pred.avg['6h']<=6 else '高') }}</td></tr>
      <tr><td>+24h</td><td>{{ last_pred.avg['24h'] }}</td><td class="{{ 'low' if last_pred.avg['24h']<=3 else ('mid' if last_pred.avg['24h']<=6 else 'hi') }}">{{ '低' if last_pred.avg['24h']<=3 else ('中' if last_pred.avg['24h']<=6 else '高') }}</td></tr>
    </table>
    <div class="ts">Generated: {{ last_pred.generated_at }}</div>
    {% else %}
    <div class="warn">⏳ No predictions yet. Trigger via POST /predict or wait for scheduler.</div>
    {% endif %}
  </div>

  <!-- API Endpoints -->
  <div class="card">
    <h2>🔌 API Endpoints</h2>
    <div class="endpoint">GET <a href="/status">/status</a> — JSON health check</div>
    <div class="endpoint">GET <a href="/predictions">/predictions</a> — Latest prediction JSON</div>
    <div class="endpoint">GET <a href="/predictions/csv">/predictions/csv</a> — Download CSV</div>
    <div class="endpoint">GET <a href="/history">/history</a> — Last 48 prediction logs</div>
    <div class="endpoint">POST /predict — Trigger pipeline now</div>
    <div class="endpoint">POST /sync — Receive hourly data (GitHub Action)</div>
  </div>
</div>

<!-- Per-district table -->
{% if last_pred %}
<div class="card">
  <h2>🗺️ Per-District Prediction (18 Areas)</h2>
  <table>
    <tr><th>District</th><th>+3h AQHI</th><th>+6h AQHI</th><th>+24h AQHI</th></tr>
    {% for d in districts %}
    <tr>
      <td>{{ d }}</td>
      <td class="{{ 'low' if last_pred.horizons['3h'][d]<=3 else ('mid' if last_pred.horizons['3h'][d]<=6 else 'hi') }}">{{ last_pred.horizons['3h'].get(d,'--') }}</td>
      <td class="{{ 'low' if last_pred.horizons['6h'][d]<=3 else ('mid' if last_pred.horizons['6h'][d]<=6 else 'hi') }}">{{ last_pred.horizons['6h'].get(d,'--') }}</td>
      <td class="{{ 'low' if last_pred.horizons['24h'][d]<=3 else ('mid' if last_pred.horizons['24h'][d]<=6 else 'hi') }}">{{ last_pred.horizons['24h'].get(d,'--') }}</td>
    </tr>
    {% endfor %}
  </table>
  <div class="ts">Colour: <span class="low">■ Low (1-3)</span>  <span class="mid">■ Medium (4-6)</span>  <span class="hi">■ High (7+)</span></div>
</div>
{% endif %}

<!-- Prediction log -->
{% if pred_log %}
<div class="card" style="margin-top:14px;">
  <h2>📜 Prediction History Log (last {{ pred_log|length }} runs)</h2>
  <table>
    <tr><th>Timestamp</th><th>Avg +3h</th><th>Avg +6h</th><th>Avg +24h</th></tr>
    {% for entry in pred_log|reverse %}
    <tr>
      <td>{{ entry.timestamp }}</td>
      <td>{{ entry.avg_3h }}</td>
      <td>{{ entry.avg_6h }}</td>
      <td>{{ entry.avg_24h }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

</body></html>
"""

# ══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def dashboard():
    """HTML dashboard — auto-refreshes every 5 min."""
    log.info("[Route] GET /  — serving dashboard")
    firebase_ok = len(firebase_admin._apps) > 0
    model_ok    = model_6h is not None
    scaler_ok   = feat_scaler is not None and aqhi_scaler is not None

    return render_template_string(
        DASHBOARD_HTML,
        server_time    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        model_ok       = model_ok,
        scaler_ok      = scaler_ok,
        firebase_ok    = firebase_ok,
        device         = str(device),
        last_pred      = last_prediction_result if last_prediction_result else None,
        pred_log       = prediction_log,
        districts      = DISTRICTS,
    )


@app.route("/status", methods=["GET"])
def status():
    """JSON health check — used by Render keep-alive and GitHub Actions."""
    log.info("[Route] GET /status")
    firebase_ok = len(firebase_admin._apps) > 0
    model_ok    = all(m is not None for m in [model_3h, model_6h, model_24h])
    scaler_ok   = feat_scaler is not None and aqhi_scaler is not None

    return jsonify({
        "status":       "healthy" if (model_ok and scaler_ok) else "degraded",
        "project":      "BuildTech GAGNN",
        "version":      "v3.0",
        "model_loaded": model_ok,
        "scalers_loaded": scaler_ok,
        "firebase_connected": firebase_ok,
        "device":       str(device),
        "last_prediction_at": last_prediction_result.get("generated_at", "never"),
        "horizons_available": [3, 6, 24],
        "server_time":  datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route("/sync", methods=["POST"])
def sync_data():
    """
    Receive one hourly data record from GitHub Action.
    Stores into sync_buffer for next pipeline run.
    """
    log.info("[Route] POST /sync — receiving data")
    try:
        data = request.json
        if not data:
            log.warning("[Sync] Empty body received")
            return jsonify({"error": "No data received"}), 400

        date_val = data.get("Date", "unknown")
        log.info(f"[Sync] ✅ Received record for Date={date_val}")
        log.info(f"[Sync]    Keys: {list(data.keys())[:10]} … ({len(data)} total)")

        # Validate it has AQHI fields
        aqhi_keys = [k for k in data.keys() if k.startswith("AQHI_")]
        log.info(f"[Sync]    AQHI keys found: {len(aqhi_keys)}")

        sync_buffer.append(data)
        if len(sync_buffer) > SEQ_LEN * 2:
            sync_buffer.pop(0)  # keep buffer manageable

        return jsonify({
            "status": "success",
            "received_date": date_val,
            "buffer_size": len(sync_buffer),
            "message": "Data buffered. Will be used in next pipeline run."
        }), 200

    except Exception as e:
        log.error(f"[Sync] ❌ Error: {e}")
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_now():
    """Manually trigger the full prediction pipeline."""
    log.info("[Route] POST /predict — manual trigger")
    try:
        result = run_prediction_pipeline()
        if result:
            return jsonify({"status": "success", "result": result}), 200
        else:
            return jsonify({"status": "error", "message": "Pipeline failed — check logs"}), 500
    except Exception as e:
        log.error(f"[Route /predict] ❌ {e}")
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Return latest prediction as JSON."""
    log.info("[Route] GET /predictions")
    if not last_prediction_result:
        log.warning("[Route] No predictions cached yet")
        # Try to read CSV fallback
        if CSV_OUTPUT_PATH.exists():
            try:
                df = pd.read_csv(CSV_OUTPUT_PATH)
                return jsonify({
                    "status":  "success (CSV fallback)",
                    "data":    df.to_dict(orient="records"),
                    "note":    "Live model cache empty; returning last saved CSV"
                }), 200
            except Exception as e:
                log.error(f"[Route] CSV read error: {e}")
        return jsonify({
            "error": "No predictions available yet. POST /predict to trigger pipeline."
        }), 404

    return jsonify({
        "status": "success",
        "data":   last_prediction_result
    }), 200


@app.route("/predictions/csv", methods=["GET"])
def get_predictions_csv():
    """Download prediction CSV."""
    log.info("[Route] GET /predictions/csv")
    if not CSV_OUTPUT_PATH.exists():
        return jsonify({"error": "CSV not found. Run /predict first."}), 404
    try:
        with open(CSV_OUTPUT_PATH, "r") as f:
            csv_content = f.read()
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gagnn_prediction_today.csv"}
        )
    except Exception as e:
        log.error(f"[Route /predictions/csv] ❌ {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    """Return rolling prediction history log."""
    log.info("[Route] GET /history")
    limit = request.args.get("limit", 48, type=int)
    return jsonify({
        "status": "success",
        "count":  len(prediction_log),
        "data":   prediction_log[-limit:]
    }), 200


@app.route("/history/csv", methods=["GET"])
def get_history_csv():
    """Download full prediction history CSV."""
    log.info("[Route] GET /history/csv")
    if not HISTORY_CSV_PATH.exists():
        return jsonify({"error": "History CSV not found yet."}), 404
    try:
        with open(HISTORY_CSV_PATH, "r") as f:
            csv_content = f.read()
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gagnn_prediction_history.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/firebase/test", methods=["GET"])
def firebase_test():
    """Test Firebase read/write connection."""
    log.info("[Route] GET /firebase/test")
    try:
        ts = datetime.now(timezone.utc).isoformat()
        firebase_db.reference("GAGNN_v3/health_check").set({
            "status": "ok",
            "tested_at": ts,
            "server": "buildtech-gnn-service.onrender.com"
        })
        val = firebase_db.reference("GAGNN_v3/health_check").get()
        log.info(f"[Firebase Test] ✅ Read back: {val}")
        return jsonify({"status": "success", "firebase_echo": val}), 200
    except Exception as e:
        log.error(f"[Firebase Test] ❌ {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Hong_Kong")

    # Run prediction pipeline at the top of every hour (HKT)
    scheduler.add_job(
        run_prediction_pipeline,
        trigger="cron",
        minute=5,   # 5 minutes past each hour (gives GitHub Action time to push data)
        id="hourly_prediction",
        replace_existing=True,
    )

    # Also run once 10 minutes after startup (cold start)
    scheduler.add_job(
        run_prediction_pipeline,
        trigger="date",
        run_date=datetime.now() + timedelta(minutes=1),
        id="startup_prediction",
    )

    scheduler.start()
    log.info("[Scheduler] ✅ APScheduler started")
    log.info("[Scheduler]    Hourly job: runs at HH:05 (HKT)")
    log.info("[Scheduler]    Startup job: runs in ~1 minute")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  BuildTech GAGNN Prediction Service  ")
    log.info("  https://buildtech-gnn-service.onrender.com/")
    log.info("=" * 60)

    # 1. Firebase
    init_firebase()

    # 2. Models + scalers + graph
    init_models()

    # 3. Background scheduler
    start_scheduler()

    # 4. Flask server
    port = int(os.environ.get("PORT", 10000))
    log.info(f"[Flask] Starting on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
