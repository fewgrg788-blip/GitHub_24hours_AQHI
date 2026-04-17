import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import re
import os
import json
import urllib.parse
from datetime import datetime, timedelta, timezone

# --- [1. 配置] ---
RENDER_GNN_API = "https://buildtech-gnn-service.onrender.com/predict"
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"  # 更改存檔名稱
HKT = timezone(timedelta(hours=8))

# AQHI 站點清單 (對應您原始 Firebase 邏輯)
STATIONS = [
    'Central_Western_General', 'Eastern_General', 'Kwun_Tong_General', 'Sham_Shui_Po_General',
    'Kwai_Chung_General', 'Tsuen_Wan_General', 'Tseung_Kwan_O_General', 'Yuen_Long_General',
    'Tuen_Mun_General', 'Tung_Chung_General', 'Tai_Po_General', 'Sha_Tin_General',
    'North_General', 'Tap_Mun_General', 'Causeway_Bay_Roadside', 'Central_Roadside',
    'Mong_Kok_Roadside', 'Southern_General'
]

# 95 欄位標準定義 (用於 CSV 存檔)
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

# 站點對應 (中文與 API ID)
STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "中環碼頭": "CP1", "青洲": "GI", "赤鱲角": "HKA",
    "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM", "流浮山": "LFS",
    "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC", "沙洲": "SC", "石壁": "SE",
    "石崗": "SEK", "九龍天星碼頭": "SF", "沙田": "SHA", "沙螺灣": "SHL", "西貢": "SKG", 
    "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME", "大埔滘": "TPK", "屯門": "TUN", 
    "大老山": "WGL", "香港濕地公園": "WLP", "香港天文台": "HKO", "九龍城": "KSC", 
    "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT", "大美督": "SSH"
}

def wind_text_to_degrees(text):
    if not text or text in ["0.0", "風向不定", "N/A"]: return None
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(text)

# --- [2. 數據獲取與核心修復函數] ---
def fetch_all_realtime():
    print("🌐 [Step 1] Fetching all real-time data...")
    fetched_data = {}
    hum_vals, wspd_vals, pdir_vals = [], [], []
    
    # 1. AQHI RSS (包含異常檢測邏輯)
    aqhi_raw = {}
    try:
        res = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=15)
        root = ET.fromstring(res.content.decode('utf-8', errors='ignore'))
        for item in root.findall(".//item"):
            desc = item.find("description").text
            title = item.find("title").text
            m = re.search(r'(\d{1,2})', re.sub(r'\d{4}', '', desc))
            key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
            key += "_Roadside" if "Roadside" in desc else "_General"
            if m: aqhi_raw[key] = int(m.group(1))

        # 均值補位邏輯
        valid_aqhis = [v for v in aqhi_raw.values() if v is not None and 0 < v <= 11]
        pre_avg = sum(valid_aqhis) / len(valid_aqhis) if valid_aqhis else 3
        
        repaired_aqhi = {}
        for s in STATIONS:
            val = aqhi_raw.get(s)
            if val is not None and abs(val - pre_avg) <= 4:
                repaired_aqhi[s] = val
            else:
                repaired_aqhi[s] = int(round(pre_avg))
        fetched_data['AQHI'] = repaired_aqhi
    except Exception as e: print(f"AQHI Fetch Error: {e}"); fetched_data['AQHI'] = {s: 3 for s in STATIONS}

    # 2. Wind & Humidity (用於 CSV 存檔)
    try:
        # Wind
        w_res = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        for line in w_res.text.strip().split('\n')[1:]:
            c = [col.strip('"') for col in line.split(',')]
            deg = wind_text_to_degrees(c[2]); spd = float(c[3]) if c[3] != "N/A" else None
            for name, sid in STATION_MAP.items():
                if name in c[1]:
                    if deg is not None: fetched_data[f"PDIR_{sid}"] = deg; pdir_vals.append(deg)
                    if spd is not None: fetched_data[f"WSPD_{sid}"] = spd; wspd_vals.append(spd)
        # Humidity
        h_res = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        for item in h_res.json().get('humidity', {}).get('data', []):
            val = float(item['value'])
            for name, sid in STATION_MAP.items():
                if name in item['place']: fetched_data[f"HUM_{sid}"] = val; hum_vals.append(val)
    except: pass

    # 計算天氣平均值備用
    fetched_data['MEANS'] = {
        "HUM": sum(hum_vals)/len(hum_vals) if hum_vals else 80.0,
        "WSPD": sum(wspd_vals)/len(wspd_vals) if wspd_vals else 5.0,
        "PDIR": sum(pdir_vals)/len(pdir_vals) if pdir_vals else 180.0
    }
    return fetched_data

# --- [3. AI 模型推理] ---
def get_gnn_prediction(current_aqhi):
    print(f"🧠 [Step 4] Requesting GNN prediction...")
    try:
        input_vector = [current_aqhi.get(s, 3) for s in STATIONS]
        response = requests.post(RENDER_GNN_API, json={"data": input_vector}, timeout=45)
        if response.status_code == 200:
            pred_data = response.json().get("prediction")
            return {STATIONS[i]: pred_data[i] for i in range(len(STATIONS))}
    except Exception as e: print(f"⚠️ GNN Error: {e}")
    return None

# --- [4. 主同步邏輯] ---
def run_integration():
    now_hkt = datetime.now(HKT)
    display_time = now_hkt.strftime("%Y-%m-%d %H:00")
    target_time_str = (now_hkt + timedelta(hours=6)).strftime("%Y-%m-%d %H:00")
    timestamp_full = now_hkt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # Firebase 初始化
    if not firebase_admin._apps:
        creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_env)), {'databaseURL': FIREBASE_URL})

    # 1. 抓取數據
    all_data = fetch_all_realtime()
    actual_aqhi = all_data['AQHI']
    current_avg = round(sum(actual_aqhi.values()) / len(STATIONS), 2)

    # 2. 更新 Firebase (保持您原有的 sync 邏輯)
    db.reference("GAGNN_24hours/GAGNN_data").update({"last_updated": timestamp_full, "readings": actual_aqhi})
    
    safe_current_path = urllib.parse.quote(display_time)
    safe_target_path = urllib.parse.quote(target_time_str)
    
    past_pred = db.reference(f"GAGNN_v2/predictions/{safe_current_path}").get()
    avg_error = 0
    if past_pred:
        errs = [abs(actual_aqhi.get(s, 0) - past_pred.get(s, 0)) for s in STATIONS]
        avg_error = sum(errs) / len(STATIONS)

    new_prediction = get_gnn_prediction(actual_aqhi)
    if new_prediction:
        db.reference(f"GAGNN_v2/predictions/{safe_target_path}").set(new_prediction)

    db.reference("GAGNN_v2/dashboard").set({
        "last_updated": display_time, "current_avg": current_avg,
        "accuracy_score": round(max(0, 100 - avg_error * 10), 2) if past_pred else 100,
        "prediction_target": target_time_str
    })

    # 3. 寫入 95 欄位資訊到 aqhi_history.csv
    row = []
    for col in ALL_COLUMNS:
        if col == "Date": row.append(now_hkt.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present": row.append(0)
        # 處理 AQHI 欄位映射
        elif col.startswith("AQHI_"):
            # 將 CSV 欄位名稱轉為 STATIONS 中的名稱
            s_key = col.replace("AQHI_", "").replace("/", "_") + "_General"
            if s_key == "Causeway Bay_General": s_key = "Causeway_Bay_Roadside"
            if s_key == "Central_General": s_key = "Central_Roadside"
            if s_key == "Mong Kok_General": s_key = "Mong_Kok_Roadside"
            # 簡化映射邏輯，直接從抓到的數據找，若找不到補平均
            val = next((v for k,v in actual_aqhi.items() if col.split('_')[1] in k.replace('_',' ')), round(current_avg))
            row.append(val)
        elif col in all_data: row.append(all_data[col])
        else:
            # 補位
            if "HUM_" in col: row.append(round(all_data['MEANS']['HUM'], 1))
            elif "WSPD_" in col: row.append(round(all_data['MEANS']['WSPD'], 1))
            elif "PDIR_" in col: row.append(round(all_data['MEANS']['PDIR'], 1))
            else: row.append(0.0)

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists: f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")

    print(f"🏁 Sync Finished. CSV updated: {CSV_FILE}")

if __name__ == "__main__":
    run_integration()
