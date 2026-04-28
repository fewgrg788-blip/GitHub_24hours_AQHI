import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import re
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta, timezone

# ====================== [Configuration] =====================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"📁 Data directory created: {DATA_DIR}")

CSV_FILE = "aqhi_history_today.csv"  # This is today's cache, kept in the root directory
# Annual history file will be: data/aqhi_history_2026.csv

FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"   # If not found, Firebase errors will be ignored

HKT = timezone(timedelta(hours=8))

AQHI_URL = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
WIND_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv"
WEATHER_JSON_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc"

# ====================== Final Enhanced STATION_MAP ======================
STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "長洲泳灘": "CCB", "中環碼頭": "CP1", "中環": "CP1", "青洲": "GI",
    "赤鱲角": "HKA", "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM",
    "流浮山": "LFS", "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC",
    "沙洲": "SC", "石壁": "SE", "石崗": "SEK", "天星碼頭": "SF", "沙田": "SHA",
    "沙螺灣": "SHL", "西貢": "SKG", "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME",
    "大埔滘": "TPK", "屯門": "TUN", "大老山": "WGL", "濕地公園": "WLP", "天文台": "HKO",
    "九龍城": "KSC", "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT", 
    "香港航海學校": "SSH", "航海學校": "SSH",
    "啟德": "KT", "赤柱": "STAN", "塔門": "TAP",

    # Official latest wind station mappings (April 2026)
    "中環碼頭": "CP1", "赤鱲角": "HKA", "長洲": "CCH", "長洲泳灘": "CCB", "青洲": "GI",
    "香港航海學校": "SSH", "啟德": "KT", "京士柏": "KP", "南丫島": "LAM", "流浮山": "LFS",
    "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "西貢": "SKG", "沙洲": "SC",
    "沙田": "SHA", "石崗": "SEK", "赤柱": "STAN", "天星碼頭": "SF", "打鼓嶺": "TKL",
    "大美督": "TME", "大埔滘": "TPK", "塔門": "TAP", "大老山": "WGL", "將軍澳": "JKB",
    "青衣": "TYW", "屯門": "TUN", "橫瀾島": "BHD", "濕地公園": "WLP", "黃竹坑": "HKS"
}

ALL_COLUMNS = [
    "Date", "AQHI_Central/Western", "AQHI_Eastern", "AQHI_Kwun Tong", "AQHI_Sham Shui Po",
    "AQHI_Kwai Chung", "AQHI_Tsuen Wan", "AQHI_Yuen Long", "AQHI_Tuen Mun", "AQHI_Tung Chung",
    "AQHI_Tai Po", "AQHI_Sha Tin", "AQHI_Tap Mun", "AQHI_Causeway Bay", "AQHI_Central",
    "AQHI_Mong Kok", "AQHI_Tseung Kwan O", "AQHI_Southern", "AQHI_North", "Cyclone_Present",
    "HUM_CCH", "HUM_HKA", "HUM_HKO", "HUM_JKB", "HUM_KP", "HUM_KSC", "HUM_LFS", "HUM_PEN",
    "HUM_SEK", "HUM_SHA", "HUM_SKG", "HUM_SSH", "HUM_TC", "HUM_TKL", "HUM_TMS", "HUM_TYW", "HUM_YCT",
    "PDIR_BHD", "PDIR_CCB", "PDIR_CCH", "PDIR_CP1", "PDIR_GI", "PDIR_HKA", "PDIR_HKS", "PDIR_JKB",
    "PDIR_KP", "PDIR_LAM", "PDIR_LFS", "PDIR_NGP", "PDIR_NP", "PDIR_PEN", "PDIR_PLC", "PDIR_SC",
    "PDIR_SE", "PDIR_SEK", "PDIR_SF", "PDIR_SHA", "PDIR_SHL", "PDIR_SKG", "PDIR_TC", "PDIR_TKL",
    "PDIR_TME", "PDIR_TPK", "PDIR_TUN", "PDIR_WGL", "PDIR_WLP", 
    "WSPD_BHD", "WSPD_CCB", "WSPD_CCH", "WSPD_CP1", "WSPD_GI", "WSPD_HKA", "WSPD_HKS", "WSPD_JKB",
    "WSPD_KP", "WSPD_LAM", "WSPD_LFS", "WSPD_NGP", "WSPD_NP", "WSPD_PEN", "WSPD_PLC", "WSPD_SC",
    "WSPD_SE", "WSPD_SEK", "WSPD_SF", "WSPD_SHA", "WSPD_SHL", "WSPD_SKG", "WSPD_TC", "WSPD_TKL",
    "WSPD_TME", "WSPD_TPK", "WSPD_TUN", "WSPD_WGL", "WSPD_WLP"
]

def wind_text_to_degrees(text):
    if not text or any(x in str(text) for x in ["0.0", "不定", "N/A", "Variable", "無風"]):
        return 0.0
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5,
               "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225,
               "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(str(text).strip(), 0.0)

def fetch_data():
    print("\n--- [🔍 API Detection Started] ---")
    fetched = {}
    risk_levels = {} # ⬅️ Store level text (e.g., "Very High")
    vals = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}

    # 1. AQHI
    try:
        r = requests.get(AQHI_URL, timeout=10)
        xml_text = re.sub(r'\sxmlns="[^"]+"', '', r.text)
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = (item.find("title").text or "")
            desc = (item.find("description").text or "")
            pure_name = title.split('-')[0].strip().replace("Roadside", "").replace("General Stations", "").strip()
            
            # 🛠️ Capture number (\d+) and level text ([a-zA-Z\s]+)
            val_match = re.search(r'(\d+)\s+([a-zA-Z\s]+)\s+-', desc, re.IGNORECASE)
            
            if val_match:
                val = int(val_match.group(1))
                level_text = val_match.group(2).strip() # Extract level text
                
                # 1. Create safe name (convert Central/Western to Central_Western)
                safe_name = pure_name.replace("/", "_")
                
                # 2. Store in risk_levels using safe name to avoid Firebase restrictions
                risk_levels[safe_name] = level_text
                
                key = f"AQHI_{pure_name}"
                if key in ALL_COLUMNS:
                    fetched[key] = val
                    vals["aqhi"].append(val)
                    print(f"✅ [AQHI] {pure_name}: {val} ({level_text})")
    except Exception as e:
        print(f"❌ AQHI Error: {e}")

    # 2. Wind CSV (precise matching)
    try:
        r = requests.get(WIND_URL, timeout=15)
        lines = r.content.decode('utf-8').strip().split('\n')
        wind_count = 0
        for line in lines[1:]:
            cols = [v.strip() for v in line.rstrip(',').split(',')]
            if len(cols) < 4: continue
            site = cols[1]
            deg = wind_text_to_degrees(cols[2])
            try: spd = float(cols[3])
            except: spd = 0.0

            matched = False
            for cn, sid in STATION_MAP.items():
                if site == cn or site.replace(" ", "") == cn.replace(" ", ""):
                    fetched[f"PDIR_{sid}"] = deg
                    fetched[f"WSPD_{sid}"] = spd
                    vals["pdir"].append(deg)
                    vals["wspd"].append(spd)
                    wind_count += 1
                    matched = True
                    break
            if not matched:
                pass # Hide unmatched wind station output for cleanliness
        print(f"✅ [Wind] Successfully fetched {wind_count} station records")
    except Exception as e:
        print(f"❌ Wind Error: {e}")

    # 3. Humidity JSON (enhanced: data broadcast)
    try:
        r = requests.get(WEATHER_JSON_URL)
        data = r.json()
        h_data = data.get('humidity', {}).get('data', [])
        
        # Get a HK-wide baseline humidity (e.g. 94%)
        global_hum = h_data[0]['value'] if h_data else 94.0
        
        # First fill all HUM columns with this baseline
        for col in ALL_COLUMNS:
            if col.startswith("HUM_"):
                fetched[col] = float(global_hum)
        
        # If API provides specific stations, override the baseline
        for item in h_data:
            place = item.get('place', '')
            val = float(item.get('value', 0))
            vals["hum"].append(val)
            for cn, sid in STATION_MAP.items():
                if cn in place:
                    fetched[f"HUM_{sid}"] = val
        print(f"✅ [Humidity Broadcast] Applied baseline humidity {global_hum}% to all monitoring stations")
    except Exception as e: print(f"❌ Humidity Error: {e}")

    # Compute baselines
    means = {
        "AQHI": round(sum(vals["aqhi"])/len(vals["aqhi"]), 1) if vals["aqhi"] else 3.0,
        "HUM": round(sum(vals["hum"])/len(vals["hum"]), 1) if vals["hum"] else 80.0,
        "WSPD": round(sum(vals["wspd"])/len(vals["wspd"]), 1) if vals["wspd"] else 8.0,
        "PDIR": 225.0
    }
    print(f"💡 [Baseline] AQHI:{means['AQHI']}, HUM:{means['HUM']}, WSPD:{means['WSPD']}")

    missing = [col for col in ALL_COLUMNS[1:] if col not in fetched and col != "Cyclone_Present"]
    print(f"⚠️ {len(missing)} columns still unmatched (expected: mostly HUM_ columns)")

    return fetched, means, risk_levels # ⬅️ Return risk_levels

# ====================== Firebase & run ======================
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print("🔥 Firebase initialized successfully")
    except:
        print("⚠️ Firebase initialization failed (ignorable)")

def upload_to_firebase(row, timestamp_str):
    try:
        ref = db.reference(f"aqhi_history/{timestamp_str.replace(' ', '_').replace(':', '-')}")
        
        # 🛠️ Fix 2: Replace "/" with "_" in ALL_COLUMNS keys to avoid Firebase errors
        safe_keys = [k.replace("/", "_") for k in ALL_COLUMNS]
        data_dict = dict(zip(safe_keys, row))
        
        ref.set(data_dict)
        print(f"✅ Firebase historical data uploaded → {timestamp_str}")
    except Exception as e:
        print(f"⚠️ Firebase historical data upload failed: {e} (ignorable)")

def save_aqhi_levels_to_firebase(risk_levels, timestamp_str):
    """
    Saves district risk level text to GAGNN_24hours/GAGNN_data/readings
    """
    if not risk_levels:
        return
    try:
        ref = db.reference("GAGNN_24hours/GAGNN_data/readings")
        data_to_save = {
            "last_update": timestamp_str,
            "station_levels": risk_levels
        }
        ref.set(data_to_save)
        print(f"✅ AQHI risk level text synced to: GAGNN_24hours/GAGNN_data/readings")
    except Exception as e:
        print(f"⚠️ Failed to sync risk level text: {e}")

def auto_wash_csv(file_path):
    if not os.path.exists(file_path):
        return
    try:
        # 1. Read and fix date format
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # 2. Normalise to the hour (e.g. 16:44 -> 16:00) and remove duplicates (keep last)
        df['Date'] = df['Date'].dt.floor('h')
        df = df.drop_duplicates(subset=['Date'], keep='last').set_index('Date')
        
        # 3. Fill in missing hours (build complete time axis)
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df = df.reindex(full_range)
        
        # 4. Smart interpolation (infer missing values from surrounding data to avoid gaps)
        df = df.interpolate(method='linear', limit_direction='both')
        
        # 5. Value constraint: AQHI must be between 1-11 and rounded to integer
        aqhi_cols = [c for c in df.columns if 'AQHI' in c]
        df[aqhi_cols] = df[aqhi_cols].clip(1, 11).round(0)
        
        # 6. Write back to CSV
        df.index.name = 'Date'
        df.reset_index().to_csv(file_path, index=False, date_format='%Y-%m-%d %H:00')
        print(f"✨ [Auto Wash] CSV cleaned, filled, and aligned to the hour")
    except Exception as e:
        print(f"⚠️ Auto Wash failed: {e}")


def fill_missing_hours_before_run(file_path):
    if not os.path.exists(file_path): 
        return
    try:
        df = pd.read_csv(file_path)
        if df.empty: return
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Remove timezone for comparison
        last_date = df['Date'].max().replace(tzinfo=None)
        now_hkt = datetime.now(HKT).replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
        if last_date < now_hkt - timedelta(hours=1):
            print(f"⚠️ Data gap detected! Last record: {last_date}, current time: {now_hkt}.")
            
            target_end = now_hkt - timedelta(hours=1)
            # Ensure no timezone when building index
            df = df.set_index('Date')
            df.index = df.index.tz_localize(None) 
            
            full_range = pd.date_range(start=df.index.min(), end=target_end, freq='h')
            df = df.reindex(full_range).ffill()
            
            aqhi_cols = [c for c in df.columns if 'AQHI' in c]
            df[aqhi_cols] = df[aqhi_cols].clip(1, 11).round(0)
            
            df.index.name = 'Date'
            df.reset_index().to_csv(file_path, index=False, date_format='%Y-%m-%d %H:00')
            print(f"✅ Gap filled successfully")
    except Exception as e:
        print(f"⚠️ Startup gap-fill check failed: {e}")


def run():
    now = datetime.now(HKT)
    timestamp_str = now.strftime("%Y-%m-%d %H:00")
    fetched, means, risk_levels = fetch_data()
    
    row = [timestamp_str]
    for col in ALL_COLUMNS[1:]:
        if col == "Cyclone_Present":
            row.append(0)
        elif col in fetched:
            row.append(fetched[col])
        else:
            if "AQHI" in col: row.append(round(means["AQHI"]))
            elif "HUM" in col: row.append(round(means["HUM"], 1))
            elif "WSPD" in col: row.append(round(means["WSPD"], 1))
            elif "PDIR" in col: row.append(means["PDIR"])
            else: row.append(0.0)

    # --- [Feature 1: Append to annual history file (path changed to data/)] ---
    current_year = now.strftime("%Y")
    # Change: path now includes DATA_DIR
    annual_history_file = os.path.join(DATA_DIR, f"aqhi_history_{current_year}.csv")
    
    history_exists = os.path.isfile(annual_history_file)
    with open(annual_history_file, "a", encoding="utf-8") as f:
        if not history_exists:
            f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")
    print(f"📦 Data synced to annual file: {annual_history_file}")

    # --- [Feature 2: Update today's cache and remove outdated data] ---
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")

    try:
        df_today = pd.read_csv(CSV_FILE)
        df_today['Date'] = pd.to_datetime(df_today['Date'])
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        df_filtered = df_today[df_today['Date'] >= today_start.replace(tzinfo=None)]
        df_filtered.to_csv(CSV_FILE, index=False, date_format='%Y-%m-%d %H:00')
        print(f"🧹 Today's cache {CSV_FILE} cleaned of outdated data")
    except Exception as e:
        print(f"⚠️ Cache cleanup failed: {e}")

    # Firebase sync
    upload_to_firebase(row, timestamp_str)
    save_aqhi_levels_to_firebase(risk_levels, timestamp_str)

def get_full_history_dataframe():
    """
    Automatically reads and merges all aqhi_history_20*.csv files from data/ directory
    """
    # Change: search within data directory
    search_path = os.path.join(DATA_DIR, "aqhi_history_20*.csv")
    all_files = sorted(glob.glob(search_path))
    print(f"📚 Reading and merging annual files: {all_files}")
    
    df_list = []
    for filename in all_files:
        df_temp = pd.read_csv(filename)
        df_list.append(df_temp)
    
    if not df_list: return pd.DataFrame()
    
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df.sort_values('Date')
    return full_df

# Recommended to encapsulate this, or ensure it is called before run()
def ensure_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"📁 Data directory confirmed: {DATA_DIR}")

if __name__ == "__main__":
    ensure_directories() # Force check
    
    current_year = datetime.now(HKT).strftime("%Y")
    target_annual_file = os.path.join(DATA_DIR, f"aqhi_history_{current_year}.csv")

    # 1. Fill data gaps
    fill_missing_hours_before_run(target_annual_file)
    
    # 2. Run fetch
    run()
    
    # 3. Clean file
    auto_wash_csv(target_annual_file)
