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
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

# Firebase 專用站點列表
STATIONS = [
    'Central_Western_General', 'Eastern_General', 'Kwun_Tong_General', 'Sham_Shui_Po_General',
    'Kwai_Chung_General', 'Tsuen_Wan_General', 'Tseung_Kwan_O_General', 'Yuen_Long_General',
    'Tuen_Mun_General', 'Tung_Chung_General', 'Tai_Po_General', 'Sha_Tin_General',
    'North_General', 'Tap_Mun_General', 'Causeway_Bay_Roadside', 'Central_Roadside',
    'Mong_Kok_Roadside', 'Southern_General'
]

# 95 欄位標準 (CSV 存檔用)
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

# 站點地圖 (比照 VSCode 版本優化)
STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "中環碼頭": "CP1", "青洲": "GI", "赤鱲角": "HKA",
    "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM", "流浮山": "LFS",
    "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC", "沙洲": "SC", "石壁": "SE",
    "石崗": "SEK", "九龍天星碼頭": "SF", "沙田": "SHA", "沙螺灣": "SHL", "西貢": "SKG", 
    "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME", "大埔滘": "TPK", "屯門": "TUN", 
    "大老山": "WGL", "香港濕地公園": "WLP", "香港天文台": "HKO", "九龍城": "KSC", 
    "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT", "大美督": "SSH",
    # AQHI Mapping
    "Central/Western": "Central/Western", "Eastern": "Eastern", "Kwun Tong": "Kwun Tong",
    "Sham Shui Po": "Sham Shui Po", "Kwai Chung": "Kwai Chung", "Tsuen Wan": "Tsuen Wan",
    "Yuen Long": "Yuen Long", "Tuen Mun": "Tuen Mun", "Tung Chung": "Tung Chung",
    "Tai Po": "Tai Po", "Sha Tin": "Sha Tin", "Tap Mun": "Tap Mun", "Causeway Bay": "Causeway Bay",
    "Central": "Central", "Mong Kok": "Mong Kok", "Tseung Kwan O": "Tseung Kwan O",
    "Southern": "Southern", "North": "North"
}

def wind_text_to_degrees(text):
    if not text or text in ["0.0", "風向不定", "N/A"]: return None
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(text)

# --- [2. 數據獲取與匹配邏輯 (VSCode 模式)] ---
def fetch_all_realtime():
    print("🌐 [Step 1] Fetching all real-time data...")
    fetched_data = {}
    aqhi_vals, hum_vals, wspd_vals, pdir_vals = [], [], [], []
    
    # 1. AQHI RSS
    try:
        res = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=15)
        root = ET.fromstring(res.content)
        aqhi_results = {}
        for item in root.findall(".//item"):
            title = item.find("title").text
            parts = title.split(':')
            if len(parts) == 2:
                s_name = parts[0].strip()
                val = int(parts[1].strip())
                if s_name in STATION_MAP:
                    aqhi_results[f"AQHI_{STATION_MAP[s_name]}"] = val
                    aqhi_vals.append(val)
        fetched_data['AQHI_RAW'] = aqhi_results
    except: fetched_data['AQHI_RAW'] = {}

    # 2. Wind CSV (精確匹配站點名稱)
    try:
        w_res = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        lines = w_res.text.strip().split('\n')
        for line in lines[1:]:
            cols = [c.strip('"').strip() for c in line.split(',')]
            if len(cols) < 4: continue
            deg = wind_text_to_degrees(cols[2])
            spd = float(cols[3]) if cols[3] != "N/A" else None
            for name, sid in STATION_MAP.items():
                if name in cols[1]: # VSCode 的 .includes() 邏輯
                    if deg is not None: 
                        fetched_data[f"PDIR_{sid}"] = deg
                        pdir_vals.append(deg)
                    if spd is not None: 
                        fetched_data[f"WSPD_{sid}"] = spd
                        wspd_vals.append(spd)
    except: pass

    # 3. Humidity JSON (精確匹配站點名稱)
    try:
        h_res = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        h_data = h_res.json().get('humidity', {}).get('data', [])
        for item in h_data:
            val = float(item['value'])
            for name, sid in STATION_MAP.items():
                if name in item['place']: # VSCode 的 .includes() 邏輯
                    fetched_data[f"HUM_{sid}"] = val
                    hum_vals.append(val)
    except: pass

    # 計算平均值 (備用補位)
    fetched_data['MEANS'] = {
        "AQHI": sum(aqhi_vals)/len(aqhi_vals) if aqhi_vals else 3,
        "HUM": sum(hum_vals)/len(hum_vals) if hum_vals else 80.0,
        "WSPD": sum(wspd_vals)/len(wspd_vals) if wspd_vals else 5.0,
        "PDIR": sum(pdir_vals)/len(pdir_vals) if pdir_vals else 0.0
    }
    return fetched_data

# --- [3. 主邏輯] ---
def run_integration():
    now_hkt = datetime.now(HKT)
    display_time = now_hkt.strftime("%Y-%m-%d %H:00")
    
    # Firebase 初始化
    if not firebase_admin._apps:
        creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_env)), {'databaseURL': FIREBASE_URL})

    # 1. 抓取數據
    data = fetch_all_realtime()
    means = data['MEANS']
    
    # 2. 準備 Firebase 需要的 18 站數據 (補位確保不報錯)
    actual_aqhi_for_firebase = {}
    for s in STATIONS:
        # 將 STATIONS 的底線名稱轉回 STATION_MAP 格式進行查找
        clean_name = s.replace('_General','').replace('_Roadside','').replace('_','/')
        val = data['AQHI_RAW'].get(f"AQHI_{clean_name}", int(round(means['AQHI'])))
        actual_aqhi_for_firebase[s] = val

    # 更新 Firebase
    db.reference("GAGNN_24hours/GAGNN_data").update({
        "last_updated": now_hkt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "readings": actual_aqhi_for_firebase
    })

    # 3. 準備 95 欄位 CSV 數據
    row = []
    for col in ALL_COLUMNS:
        if col == "Date": row.append(now_hkt.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present": row.append(0)
        elif col in data: row.append(data[col]) # 優先使用實時抓到的值
        else:
            # 補位邏輯
            if col.startswith("AQHI_"): row.append(round(means['AQHI']))
            elif col.startswith("HUM_"): row.append(round(means['HUM'], 1))
            elif col.startswith("WSPD_"): row.append(round(means['WSPD'], 1))
            elif col.startswith("PDIR_"): row.append(round(means['PDIR'], 1))
            else: row.append(0.0)

    # 寫入 CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists: f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")

    print(f"🏁 Sync Finished. CSV updated with Real-time metrics.")

if __name__ == "__main__":
    run_integration()
