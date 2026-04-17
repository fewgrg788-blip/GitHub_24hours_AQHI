import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import re
from datetime import datetime, timedelta, timezone

# --- [1. 配置] ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

# 95 欄位定義 (略, 保持不變)
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

STATIONS_FIREBASE = [
    'Central_Western_General', 'Eastern_General', 'Kwun_Tong_General', 'Sham_Shui_Po_General',
    'Kwai_Chung_General', 'Tsuen_Wan_General', 'Tseung_Kwan_O_General', 'Yuen_Long_General',
    'Tuen_Mun_General', 'Tung_Chung_General', 'Tai_Po_General', 'Sha_Tin_General',
    'North_General', 'Tap_Mun_General', 'Causeway_Bay_Roadside', 'Central_Roadside',
    'Mong_Kok_Roadside', 'Southern_General'
]

# 核心映射表 (已根據官方 API 格式優化)
STATION_MAP = {
    "橫瀾島": "BHD", "Waglan Island": "BHD", "長洲": "CCH", "Cheung Chau": "CCH",
    "中環": "CP1", "Central": "CP1", "赤鱲角": "HKA", "Chek Lap Kok": "HKA",
    "黃竹坑": "HKS", "Wong Chuk Hang": "HKS", "將軍澳": "JKB", "Tseung Kwan O": "JKB",
    "京士柏": "KP", "King's Park": "KP", "流浮山": "LFS", "Lau Fau Shan": "LFS",
    "昂坪": "NGP", "Ngong Ping": "NGP", "北角": "NP", "North Point": "NP",
    "沙田": "SHA", "Sha Tin": "SHA", "西貢": "SKG", "Sai Kung": "SKG",
    "東涌": "TC", "Tung Chung": "TC", "打鼓嶺": "TKL", "Ta Kwu Ling": "TKL",
    "大美督": "TME", "Tai Mei Tuk": "TME", "屯門": "TUN", "Tuen Mun": "TUN",
    "大老山": "WGL", "Tate's Cairn": "WGL", "濕地公園": "WLP", "Wetland Park": "WLP",
    "天文台": "HKO", "Observatory": "HKO", "九龍城": "KSC", "Kowloon City": "KSC",
    "大帽山": "TMS", "Tai Mo Shan": "TMS", "青衣": "TYW", "Tsing Yi": "TYW", "元朗": "YCT", "Yuen Long": "YCT"
}

def wind_text_to_degrees(text):
    if not text or any(x in text for x in ["0.0", "不定", "N/A", "Variable"]): return None
    mapping = {"北": 0, "N": 0, "東北": 45, "NE": 45, "東": 90, "E": 90, "東南": 135, "SE": 135, "南": 180, "S": 180, "西南": 225, "SW": 225, "西": 270, "W": 270, "西北": 315, "NW": 315}
    # 模糊匹配開頭
    for k, v in mapping.items():
        if text.startswith(k): return v
    return None

def fetch_with_deep_debug():
    print("\n--- 🔍 開始深度數據抓取 (DEBUG MODE) ---")
    fetched = {}
    v = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}

    # 1. AQHI RSS
    try:
        r = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=15)
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            title = item.find("title").text # "Central/Western: 3"
            if ":" in title:
                loc_raw, val_raw = title.split(":")
                loc_name = loc_raw.strip()
                try:
                    val = int(re.search(r'\d+', val_raw).group())
                    fetched[f"AQHI_{loc_name}"] = val
                    v["aqhi"].append(val)
                except: pass
        print(f"✅ AQHI: 成功匹配 {len(v['aqhi'])} 個站點 (包含 {list(fetched.keys())[:3]}...)")
    except Exception as e: print(f"❌ AQHI 錯誤: {e}")

    # 2. Wind CSV
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        lines = r.text.strip().split('\n')[1:]
        for line in lines:
            c = [x.strip('"').strip() for x in line.split(',')]
            if len(c) < 4: continue
            # 遍歷 Map 進行模糊匹配
            for name, sid in STATION_MAP.items():
                if name in c[1]:
                    deg = wind_text_to_degrees(c[2])
                    try: spd = float(c[3])
                    except: spd = None
                    if deg is not None: fetched[f"PDIR_{sid}"] = deg; v["pdir"].append(deg)
                    if spd is not None: fetched[f"WSPD_{sid}"] = spd; v["wspd"].append(spd)
    except Exception as e: print(f"❌ 風力錯誤: {e}")

    # 3. Humidity JSON
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        h_list = r.json().get('humidity', {}).get('data', [])
        for item in h_list:
            place = item['place']
            val = float(item['value'])
            for name, sid in STATION_MAP.items():
                if name in place:
                    fetched[f"HUM_{sid}"] = val
                    v["hum"].append(val)
                    break
        print(f"✅ 濕度: 成功匹配 {len(v['hum'])} 個站點。")
    except Exception as e: print(f"❌ 濕度錯誤: {e}")

    means = {
        "AQHI": sum(v["aqhi"])/len(v["aqhi"]) if v["aqhi"] else 3.0,
        "HUM": sum(v["hum"])/len(v["hum"]) if v["hum"] else 80.0,
        "WSPD": sum(v["wspd"])/len(v["wspd"]) if v["wspd"] else 5.0,
        "PDIR": sum(v["pdir"])/len(v["pdir"]) if v["pdir"] else 180.0
    }
    return fetched, means

def run_sync():
    now_hkt = datetime.now(HKT)
    fetched, means = fetch_with_deep_debug()

    # Firebase 更新 (18 站)
    try:
        if not firebase_admin._apps:
            creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_env)), {'databaseURL': FIREBASE_URL})
        
        fb_readings = {}
        for s in STATIONS_FIREBASE:
            # 轉換名稱以匹配 fetched 鍵值
            short_name = s.replace('_General','').replace('_Roadside','').replace('_',' ')
            # 嘗試精確匹配或包含匹配
            val = None
            for k, v in fetched.items():
                if "AQHI_" in k and short_name in k.replace('/',' '):
                    val = v
                    break
            fb_readings[s] = val if val is not None else int(round(means["AQHI"]))
        
        db.reference("GAGNN_24hours/GAGNN_data").update({
            "last_updated": now_hkt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "readings": fb_readings
        })
    except Exception as e: print(f"⚠️ Firebase 失敗: {e}")

    # CSV 95 欄位寫入
    row = []
    real_c = 0
    for col in ALL_COLUMNS:
        if col == "Date": row.append(now_hkt.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present": row.append(0)
        elif col in fetched:
            row.append(fetched[col]); real_c += 1
        else:
            # 補位邏輯
            if "AQHI" in col:
                # 再次嘗試模糊查找 AQHI
                base = col.replace("AQHI_","")
                val = next((v for k,v in fetched.items() if base in k), round(means["AQHI"]))
                row.append(val)
                if val != round(means["AQHI"]): real_c += 1
            elif "HUM" in col: row.append(round(means["HUM"], 1))
            elif "WSPD" in col: row.append(round(means["WSPD"], 1))
            elif "PDIR" in col: row.append(round(means["PDIR"], 1))
            else: row.append(0.0)

    print(f"📊 最終存檔報告: [真實值: {real_c}] [預設值: {95-real_c}] [健康度: {(real_c/95)*100:.1f}%]")

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists: f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")

if __name__ == "__main__":
    run_sync()
