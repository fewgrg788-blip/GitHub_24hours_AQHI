import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import json
import re
from datetime import datetime, timedelta, timezone

# --- [1. 配置] ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

# 站點對應表 (完全對齊 JS 版本)
STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "中環碼頭": "CP1", "青洲": "GI", "赤鱲角": "HKA",
    "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM", "流浮山": "LFS",
    "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC", "沙洲": "SC", "石壁": "SE",
    "石崗": "SEK", "九龍天星碼頭": "SF", "沙田": "SHA", "沙螺灣": "SHL", "西貢": "SKG", 
    "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME", "大埔滘": "TPK", "屯門": "TUN", 
    "大老山": "WGL", "香港濕地公園": "WLP", "香港天文台": "HKO", "九龍城": "KSC", 
    "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT", 
    # AQHI 專用映射
    "Central/Western": "Central/Western", "Eastern": "Eastern", "Kwun Tong": "Kwun Tong",
    "Sham Shui Po": "Sham Shui Po", "Kwai Chung": "Kwai Chung", "Tsuen Wan": "Tsuen Wan",
    "Yuen Long": "Yuen Long", "Tuen Mun": "Tuen Mun", "Tung Chung": "Tung Chung",
    "Tai Po": "Tai Po", "Sha Tin": "Sha Tin", "Tap Mun": "Tap Mun", "Causeway Bay": "Causeway Bay",
    "Central": "Central", "Mong Kok": "Mong Kok", "Tseung Kwan O": "Tseung Kwan O",
    "Southern": "Southern", "North": "North"
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
    if not text or text in ["0.0", "風向不定", "N/A", "Variable"]: return None
    mapping = {
        "北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "東南": 135, "南南東": 157.5,
        "南": 180, "南南西": 202.5, "西南": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5
    }
    return mapping.get(text)

def calc_mean(arr):
    return round(sum(arr) / len(arr), 1) if arr else 0.0

# --- [2. 核心抓取函數] ---
def fetch_data():
    print("🚀 開始抓取數據 (Python Debug Mode)...")
    fetched = {}
    vals = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}

    # 1. AQHI RSS
    try:
        r = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=10)
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        for item in items:
            title = item.find("title").text # "Central/Western: 3"
            parts = title.split(':')
            if len(parts) == 2:
                station = parts[0].strip()
                try:
                    val = int(parts[1].strip())
                    if station in STATION_MAP:
                        mapped_id = STATION_MAP[station]
                        fetched[f"AQHI_{mapped_id}"] = val
                        vals["aqhi"].append(val)
                    else:
                        print(f"⚠️ AQHI 站點未在 Map 中: [{station}]")
                except: pass
    except Exception as e: print(f"❌ AQHI RSS 失敗: {e}")

    # 2. Wind CSV
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        lines = r.text.strip().split('\n')[1:]
        for line in lines:
            cols = [v.replace('"', '').strip() for v in line.split(',')]
            if len(cols) < 4: continue
            
            pdir = wind_text_to_degrees(cols[2])
            try: wspd = float(cols[3])
            except: wspd = None

            for name, sid in STATION_MAP.items():
                if name in cols[1]: # JS: cols[1].includes(name)
                    if pdir is not None:
                        fetched[f"PDIR_{sid}"] = pdir
                        vals["pdir"].append(pdir)
                    if wspd is not None:
                        fetched[f"WSPD_{sid}"] = wspd
                        vals["wspd"].append(wspd)
    except Exception as e: print(f"❌ 風力 CSV 失敗: {e}")

    # 3. Humidity JSON
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        hum_data = r.json()['humidity']['data']
        for item in hum_data:
            val = float(item['value'])
            for name, sid in STATION_MAP.items():
                if name in item['place']:
                    fetched[f"HUM_{sid}"] = val
                    vals["hum"].append(val)
    except Exception as e: print(f"❌ 濕度 JSON 失敗: {e}")

    means = {
        "AQHI": calc_mean(vals["aqhi"]) or 3.0,
        "HUM": calc_mean(vals["hum"]) or 80.0,
        "WSPD": calc_mean(vals["wspd"]) or 5.0,
        "PDIR": calc_mean(vals["pdir"]) or 0.0
    }
    return fetched, means

# --- [3. 執行同步] ---
def run():
    now = datetime.now(HKT)
    fetched, means = fetch_data()

    # A. Firebase 同步 (修正 json.loads 報錯)
    try:
        if not firebase_admin._apps:
            creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            if creds_json:
                # 這裡確保 json 已導入
                creds_dict = json.loads(creds_json)
                firebase_admin.initialize_app(credentials.Certificate(creds_dict), {'databaseURL': FIREBASE_URL})
                
                # 更新 Firebase 讀數
                fb_data = {}
                for col in ALL_COLUMNS:
                    if col.startswith("AQHI_"):
                        # 去掉 AQHI_ 前綴並將 / 換回 _ 以符合 Firebase 格式
                        key = col.replace("AQHI_", "").replace("/", "_") + "_General"
                        fb_data[key] = fetched.get(col, int(round(means["AQHI"])))
                
                db.reference("GAGNN_24hours/GAGNN_data").update({
                    "last_updated": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "readings": fb_data
                })
                print("✅ Firebase 同步成功")
    except Exception as e:
        print(f"⚠️ Firebase 錯誤: {e}")

    # B. CSV 寫入
    row = []
    real_count = 0
    for col in ALL_COLUMNS:
        if col == "Date": row.append(now.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present": row.append(0)
        elif col in fetched:
            row.append(fetched[col])
            real_count += 1
        else:
            if "AQHI" in col: row.append(round(means["AQHI"]))
            elif "HUM" in col: row.append(means["HUM"])
            elif "WSPD" in col: row.append(means["WSPD"])
            elif "PDIR" in col: row.append(means["PDIR"])
            else: row.append(0.0)

    print(f"📊 報告: 真實匹配={real_count}, 健康度={(real_count/95)*100:.1f}%")
    
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        f.write(",".join(map(str, row)) + "\n")

if __name__ == "__main__":
    run()
