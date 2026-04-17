import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import os
import json
from datetime import datetime, timedelta, timezone

# --- [1. 配置] ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

STATIONS = [
    'Central_Western_General', 'Eastern_General', 'Kwun_Tong_General', 'Sham_Shui_Po_General',
    'Kwai_Chung_General', 'Tsuen_Wan_General', 'Tseung_Kwan_O_General', 'Yuen_Long_General',
    'Tuen_Mun_General', 'Tung_Chung_General', 'Tai_Po_General', 'Sha_Tin_General',
    'North_General', 'Tap_Mun_General', 'Causeway_Bay_Roadside', 'Central_Roadside',
    'Mong_Kok_Roadside', 'Southern_General'
]

# 站點地圖 (修正與天文台 API 匹配的關鍵字)
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
    if not text or any(x in text for x in ["0.0", "不定", "N/A"]): return None
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(text)

# --- [2. 核心抓取函數] ---
def fetch_realtime_precise():
    fetched = {}
    vals = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}
    
    # 1. AQHI
    try:
        r = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=10)
        items = ET.fromstring(r.content).findall(".//item")
        for it in items:
            t = it.find("title").text
            if ":" in t:
                loc, val = t.split(":")
                loc = loc.strip()
                if loc in ["Central/Western", "Eastern", "Kwun Tong", "Sham Shui Po", "Kwai Chung", "Tsuen Wan", "Yuen Long", "Tuen Mun", "Tung Chung", "Tai Po", "Sha Tin", "Tap Mun", "Causeway Bay", "Central", "Mong Kok", "Tseung Kwan O", "Southern", "North"]:
                    fetched[f"AQHI_{loc}"] = int(val.strip())
                    vals["aqhi"].append(int(val.strip()))
    except: pass

    # 2. Wind (修正匹配邏輯)
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        lines = r.text.strip().split('\n')[1:]
        for line in lines:
            c = [x.strip('"').strip() for x in line.split(',')]
            if len(c) < 4: continue
            # 遍歷 Map 找到匹配的站點
            for name, sid in STATION_MAP.items():
                if name in c[1]: # 只要 API 的站點名包含地圖裡的關鍵字
                    deg = wind_text_to_degrees(c[2])
                    try: spd = float(c[3])
                    except: spd = None
                    if deg is not None: 
                        fetched[f"PDIR_{sid}"] = deg
                        vals["pdir"].append(deg)
                    if spd is not None: 
                        fetched[f"WSPD_{sid}"] = spd
                        vals["wspd"].append(spd)
    except: pass

    # 3. Humidity
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        for item in r.json()['humidity']['data']:
            for name, sid in STATION_MAP.items():
                if name in item['place']:
                    val = float(item['value'])
                    fetched[f"HUM_{sid}"] = val
                    vals["hum"].append(val)
    except: pass

    # 計算平均值
    means = {
        "AQHI": sum(vals["aqhi"])/len(vals["aqhi"]) if vals["aqhi"] else 3,
        "HUM": sum(vals["hum"])/len(vals["hum"]) if vals["hum"] else 80.0,
        "WSPD": sum(vals["wspd"])/len(vals["wspd"]) if vals["wspd"] else 5.0,
        "PDIR": sum(vals["pdir"])/len(vals["pdir"]) if vals["pdir"] else 0.0
    }
    return fetched, means

def run():
    now = datetime.now(HKT)
    data, means = fetch_realtime_precise()
    
    # 寫入 CSV (95 欄位)
    from sync_wan_data import ALL_COLUMNS # 引用之前的欄位定義
    row = []
    for col in ALL_COLUMNS:
        if col == "Date": row.append(now.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present": row.append(0)
        elif col in data: row.append(data[col])
        else:
            # 補位
            if "AQHI" in col: row.append(round(means["AQHI"]))
            elif "HUM" in col: row.append(round(means["HUM"], 1))
            elif "WSPD" in col: row.append(round(means["WSPD"], 1))
            elif "PDIR" in col: row.append(round(means["PDIR"], 1))
            else: row.append(0.0)
            
    with open(CSV_FILE, "a") as f:
        f.write(",".join(map(str, row)) + "\n")
    print(f"✅ CSV Updated: {now.strftime('%H:%M')}, HUM: {means['HUM']}, PDIR Sample: {means['PDIR']}")

if __name__ == "__main__":
    run()
