import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import json
import re
from datetime import datetime, timedelta, timezone

# --- [配置] ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "長洲泳灘": "CCB", "中環": "CP1", "青洲": "GI", 
    "赤鱲角": "HKA", "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM", 
    "流浮山": "LFS", "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC", 
    "沙洲": "SC", "石壁": "SE", "石崗": "SEK", "天星碼頭": "SF", "沙田": "SHA", 
    "沙螺灣": "SHL", "西貢": "SKG", "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME", 
    "大埔滘": "TPK", "屯門": "TUN", "大老山": "WGL", "濕地公園": "WLP", "天文台": "HKO", 
    "九龍城": "KSC", "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT", "大美督": "SSH",
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
    if not text or any(x in text for x in ["0.0", "不定", "N/A", "Variable", "無風"]): return 0.0
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(text, 0.0)

def fetch_data():
    print("🚀 開始數據抓取...")
    fetched = {}
    vals = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}

    try:
        r = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=10)
        content = re.sub(r'\sxmlns="[^"]+"', '', r.text, count=1)
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title = item.find("title").text
            if ":" in title:
                st_name, val_str = [x.strip() for x in title.split(":", 1)]
                val = int(re.search(r'\d+', val_str).group())
                clean_st = st_name.replace(" ", "")
                for k, sid in STATION_MAP.items():
                    if k.replace(" ", "") == clean_st:
                        fetched[f"AQHI_{sid}"] = val
                        vals["aqhi"].append(val)
    except: pass

    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        csv_text = r.content.decode('utf-8')
        lines = csv_text.strip().split('\n')
        for line in lines[1:]:
            cols = [v.replace('"', '').strip() for v in line.split(',')]
            if len(cols) < 4: continue
            for name, sid in STATION_MAP.items():
                if name in cols[1]:
                    deg = wind_text_to_degrees(cols[2])
                    try: spd = float(cols[3])
                    except: spd = 0.0
                    fetched[f"PDIR_{sid}"] = deg; vals["pdir"].append(deg)
                    fetched[f"WSPD_{sid}"] = spd; vals["wspd"].append(spd)
    except: pass

    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        for item in r.json().get('humidity', {}).get('data', []):
            for name, sid in STATION_MAP.items():
                if name in item['place']:
                    val = float(item['value'])
                    fetched[f"HUM_{sid}"] = val
                    vals["hum"].append(val)
    except: pass

    # --- 關鍵修正處：統一使用大寫鍵名 ---
    means = {
        "AQHI": sum(vals["aqhi"])/len(vals["aqhi"]) if vals["aqhi"] else 3.0,
        "HUM": sum(vals["hum"])/len(vals["hum"]) if vals["hum"] else 75.0,
        "WSPD": sum(vals["wspd"])/len(vals["wspd"]) if vals["wspd"] else 5.0,
        "PDIR": sum(vals["pdir"])/len(vals["pdir"]) if vals["pdir"] else 0.0
    }
    return fetched, means

def run():
    now = datetime.now(HKT)
    fetched, means = fetch_data()

    # Firebase 更新
    try:
        if not firebase_admin._apps:
            creds = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
            firebase_admin.initialize_app(credentials.Certificate(creds), {'databaseURL': FIREBASE_URL})
        fb_readings = {}
        for col in ALL_COLUMNS:
            if col.startswith("AQHI_"):
                key = col.replace("AQHI_", "").replace("/", "_") + "_General"
                fb_readings[key] = fetched.get(col, int(round(means["AQHI"])))
        db.reference("GAGNN_24hours/GAGNN_data").update({
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S"),
            "readings": fb_readings
        })
    except: pass

    # CSV 寫入
    row = []
    real_count = 0
    for col in ALL_COLUMNS:
        if col == "Date": row.append(now.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present": row.append(0)
        elif col in fetched:
            row.append(fetched[col]); real_count += 1
        else:
            if "AQHI" in col: row.append(round(means["AQHI"]))
            elif "HUM" in col: row.append(round(means["HUM"], 1))
            elif "WSPD" in col: row.append(round(means["WSPD"], 1))
            elif "PDIR" in col: row.append(round(means["PDIR"], 1))
            else: row.append(0.0)

    print(f"📊 匹配完成！真實數據點: {real_count} / 95")
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        f.write(",".join(map(str, row)) + "\n")

if __name__ == "__main__":
    run()
