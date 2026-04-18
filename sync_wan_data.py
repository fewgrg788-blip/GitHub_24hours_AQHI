import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import re
import pandas as pd
from datetime import datetime, timedelta, timezone

# ====================== [Configuration] ======================
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

AQHI_URL = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
WIND_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv"
WEATHER_JSON_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc"

STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "長洲泳灘": "CCB", "中環碼頭": "CP1", "中環": "CP1", "青洲": "GI",
    "赤鱲角": "HKA", "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM",
    "流浮山": "LFS", "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC",
    "沙洲": "SC", "石壁": "SE", "石崗": "SEK", "天星碼頭": "SF", "沙田": "SHA",
    "沙螺灣": "SHL", "西貢": "SKG", "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME",
    "大埔滘": "TPK", "屯門": "TUN", "大老山": "WGL", "濕地公園": "WLP", "天文台": "HKO",
    "九龍城": "KSC", "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT", 
    "香港航海學校": "SSH", "航海學校": "SSH", "啟德": "KT", "赤柱": "STAN", "塔門": "TAP"
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
    if not text or any(x in str(text) for x in ["0.0", "不定", "N/A", "Variable", "無風"]): return 0.0
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5,
               "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225,
               "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(str(text).strip(), 0.0)

def fetch_data():
    fetched, risk_levels = {}, {}
    vals = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}
    
    # 1. AQHI Data
    try:
        r = requests.get(AQHI_URL, timeout=10)
        root = ET.fromstring(re.sub(r'\sxmlns="[^"]+"', '', r.text))
        for item in root.findall(".//item"):
            title = (item.find("title").text or "")
            desc = (item.find("description").text or "")
            pure_name = title.split('-')[0].strip().replace("Roadside", "").replace("General Stations", "").strip()
            val_match = re.search(r'(\d+)\s+([a-zA-Z\s]+)\s+-', desc, re.IGNORECASE)
            if val_match:
                val = int(val_match.group(1))
                fetched[f"AQHI_{pure_name}"] = val
                safe_key = pure_name.replace("/", "_")
                risk_levels[safe_key] = val_match.group(2).strip()
                vals["aqhi"].append(val)
    except: print("⚠️ AQHI Fetch Failed")

    # 2. Wind Data
    try:
        r = requests.get(WIND_URL, timeout=15)
        lines = r.content.decode('utf-8').strip().split('\n')
        for line in lines[1:]:
            cols = [v.strip() for v in line.rstrip(',').split(',')]
            if len(cols) < 4: continue
            for cn, sid in STATION_MAP.items():
                if cols[1] == cn:
                    fetched[f"PDIR_{sid}"] = wind_text_to_degrees(cols[2])
                    fetched[f"WSPD_{sid}"] = float(cols[3])
                    vals["wspd"].append(float(cols[3]))
                    vals["pdir"].append(wind_text_to_degrees(cols[2]))
                    break
    except: print("⚠️ Wind Fetch Failed")

    # 3. Humidity Data
    try:
        r = requests.get(WEATHER_JSON_URL)
        h_data = r.json().get('humidity', {}).get('data', [])
        global_hum = float(h_data[0]['value']) if h_data else 80.0
        for col in ALL_COLUMNS:
            if col.startswith("HUM_"): fetched[col] = global_hum
        for item in h_data:
            for cn, sid in STATION_MAP.items():
                if cn in item.get('place', ''):
                    fetched[f"HUM_{sid}"] = float(item.get('value', global_hum))
    except: print("⚠️ Humidity Fetch Failed")
    
    means = {
        "AQHI": sum(vals["aqhi"])/len(vals["aqhi"]) if vals["aqhi"] else 3.0,
        "HUM": 80.0,
        "WSPD": sum(vals["wspd"])/len(vals["wspd"]) if vals["wspd"] else 5.0,
        "PDIR": 225.0
    }
    return fetched, means, risk_levels

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
    except: print("⚠️ Firebase Init Failed")

def auto_clean_and_align_csv(file_path):
    try:
        if not os.path.exists(file_path): return
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.floor('H')
        df = df.dropna(subset=['Date']).drop_duplicates(subset=['Date'], keep='last').set_index('Date')
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
        df_aligned = df.reindex(full_range).ffill().bfill()
        df_aligned.reset_index().rename(columns={'index': 'Date'}).to_csv(file_path, index=False)
        print("✅ CSV Aligned to Hourly Frequency.")
    except Exception as e: print(f"❌ Clean Error: {e}")

def run():
    now = datetime.now(HKT)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    fetched, means, risk_levels = fetch_data()
    
    row = [timestamp_str]
    for col in ALL_COLUMNS[1:]:
        if col == "Cyclone_Present": row.append(0)
        elif col in fetched: row.append(fetched[col])
        else:
            m_key = col.split('_')[0]
            row.append(round(means.get(m_key, 0), 1))

    # Header Integrity Check
    file_exists = os.path.isfile(CSV_FILE)
    reset_file = False
    if file_exists:
        try:
            if len(pd.read_csv(CSV_FILE, nrows=0).columns) != len(ALL_COLUMNS): reset_file = True
        except: reset_file = True

    with open(CSV_FILE, "w" if (not file_exists or reset_file) else "a", encoding="utf-8") as f:
        if not file_exists or reset_file: f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")
    
    # Firebase Upload with Key Sanitization
    safe_data = {k.replace("/", "_"): v for k, v in zip(ALL_COLUMNS, row)}
    try:
        db.reference(f"aqhi_history/{timestamp_str.replace(' ', '_').replace(':', '-')}") \
          .set(safe_data)
        db.reference("GAGNN_24hours/GAGNN_data/readings") \
          .set({"last_update": timestamp_str, "station_levels": risk_levels})
        print(f"✅ Firebase Sync Success: {timestamp_str}")
    except Exception as e: print(f"❌ Firebase Error: {e}")

if __name__ == "__main__":
    run()
    auto_clean_and_align_csv(CSV_FILE)
