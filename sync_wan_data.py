import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import re
from datetime import datetime, timedelta, timezone

# ====================== [配置] ======================
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"   # 沒有就忽略 Firebase 錯誤

CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

AQHI_URL = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
WIND_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv"
WEATHER_JSON_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc"

# ====================== 最終加強版 STATION_MAP ======================
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

    # 官方最新風速站點精準對應（2026年4月）
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
    print("\n--- [🔍 API 檢測開始] ---")
    fetched = {}
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
            
            # 🛠️ 修復 1：加入 re.IGNORECASE，讓它無視大小寫，完美捕捉 "Very high"
            val_match = re.search(r'(\d+)\s+(Low|Moderate|High|Very High|Serious)', desc, re.IGNORECASE)
            
            if val_match:
                val = int(val_match.group(1))
                key = f"AQHI_{pure_name}"
                if key in ALL_COLUMNS:
                    fetched[key] = val
                    vals["aqhi"].append(val)
                    print(f"✅ [AQHI] {pure_name}: {val}")
    except Exception as e:
        print(f"❌ AQHI 錯誤: {e}")

    # 2. Wind CSV（精準匹配）
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
                print(f"   未匹配風站: {site}")
        print(f"✅ [風力] 已成功抓取 {wind_count} 個站點數據")
    except Exception as e:
        print(f"❌ Wind 錯誤: {e}")

    # 3. Humidity JSON (強化版：數據廣播)
    try:
        r = requests.get(WEATHER_JSON_URL)
        data = r.json()
        h_data = data.get('humidity', {}).get('data', [])
        
        # 獲取一個全港通用的基準濕度 (例如 94%)
        global_hum = h_data[0]['value'] if h_data else 94.0
        
        # 先用這個基準值填滿所有 HUM 欄位
        for col in ALL_COLUMNS:
            if col.startswith("HUM_"):
                fetched[col] = float(global_hum)
        
        # 如果 API 有提供特定站點，再進行覆蓋
        for item in h_data:
            place = item.get('place', '')
            val = float(item.get('value', 0))
            vals["hum"].append(val)
            for cn, sid in STATION_MAP.items():
                if cn in place:
                    fetched[f"HUM_{sid}"] = val
        print(f"✅ [濕度廣播] 已將基準濕度 {global_hum}% 應用於所有濕度監測點")
    except Exception as e: print(f"❌ Humidity 錯誤: {e}")

    # 計算基準
    means = {
        "AQHI": round(sum(vals["aqhi"])/len(vals["aqhi"]), 1) if vals["aqhi"] else 3.0,
        "HUM": round(sum(vals["hum"])/len(vals["hum"]), 1) if vals["hum"] else 80.0,
        "WSPD": round(sum(vals["wspd"])/len(vals["wspd"]), 1) if vals["wspd"] else 8.0,
        "PDIR": 225.0
    }
    print(f"💡 [計算基準] AQHI:{means['AQHI']}, HUM:{means['HUM']}, WSPD:{means['WSPD']}")

    missing = [col for col in ALL_COLUMNS[1:] if col not in fetched and col != "Cyclone_Present"]
    print(f"⚠️ 仍有 {len(missing)} 個欄位未匹配（預期：主要是 HUM_ 欄位）")

    return fetched, means

# ====================== Firebase & run ======================
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print("🔥 Firebase 初始化成功")
    except:
        print("⚠️ Firebase 初始化失敗（可忽略）")

def upload_to_firebase(row, timestamp_str):
    try:
        ref = db.reference(f"aqhi_history/{timestamp_str.replace(' ', '_').replace(':', '-')}")
        
        # 🛠️ 修復 2：將 ALL_COLUMNS 裡面的 "/" 替換成 "_"，避免 Firebase 報錯
        safe_keys = [k.replace("/", "_") for k in ALL_COLUMNS]
        data_dict = dict(zip(safe_keys, row))
        
        ref.set(data_dict)
        print(f"✅ Firebase 上傳成功 → {timestamp_str}")
    except Exception as e:
        print(f"⚠️ Firebase 上傳失敗: {e}（可忽略）")

def run():
    now = datetime.now(HKT)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    fetched, means = fetch_data()
    
    row = [timestamp_str]
    matched = 0
    
    for col in ALL_COLUMNS[1:]:
        if col == "Cyclone_Present":
            row.append(0)
        elif col in fetched:
            row.append(fetched[col])
            matched += 1
        else:
            if "AQHI" in col: row.append(round(means["AQHI"]))
            elif "HUM" in col: row.append(round(means["HUM"], 1))
            elif "WSPD" in col: row.append(round(means["WSPD"], 1))
            elif "PDIR" in col: row.append(means["PDIR"])
            else: row.append(0.0)

    # 寫入 CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")
    
    upload_to_firebase(row, timestamp_str)

    print(f"\n--- [📊 執行完成] ---")
    print(f"時間: {timestamp_str}")
    print(f"匹配成功: {matched} / {len(ALL_COLUMNS)-1} 欄位")
    print(f"CSV 已更新: {CSV_FILE}")

if __name__ == "__main__":
    run()
