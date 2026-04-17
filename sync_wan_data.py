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

# 95 欄位定義
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

STATION_MAP = {
    "橫瀾島": "BHD", "長洲": "CCH", "中環碼頭": "CP1", "青洲": "GI", "赤鱲角": "HKA",
    "黃竹坑": "HKS", "將軍澳": "JKB", "京士柏": "KP", "南丫島": "LAM", "流浮山": "LFS",
    "昂坪": "NGP", "北角": "NP", "坪洲": "PEN", "山頂": "PLC", "沙洲": "SC", "石壁": "SE",
    "石崗": "SEK", "九龍天星碼頭": "SF", "沙田": "SHA", "沙螺灣": "SHL", "西貢": "SKG", 
    "東涌": "TC", "打鼓嶺": "TKL", "大美督": "TME", "大埔滘": "TPK", "屯門": "TUN", 
    "大老山": "WGL", "香港濕地公園": "WLP", "香港天文台": "HKO", "九龍城": "KSC", 
    "大帽山": "TMS", "青衣": "TYW", "元朗": "YCT"
}

def wind_text_to_degrees(text):
    if not text or any(x in text for x in ["0.0", "不定", "N/A"]): return None
    mapping = {"北": 0, "北北東": 22.5, "東北": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "東南": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "西南": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "西北": 315, "北西北": 337.5}
    return mapping.get(text)

# --- [2. 核心抓取邏輯 + Debug 追蹤] ---
def fetch_with_debug():
    print("\n--- 🔍 正在抓取實時數據並進行調試分析 ---")
    fetched = {}
    source_track = {"Real": 0, "Default": 0}
    v = {"aqhi": [], "hum": [], "wspd": [], "pdir": []}

    # 1. AQHI RSS
    try:
        r = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml", timeout=10)
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            title = item.find("title").text
            if ":" in title:
                loc, val = title.split(":")
                loc, val = loc.strip(), int(val.strip())
                fetched[f"AQHI_{loc}"] = val
                v["aqhi"].append(val)
        print(f"✅ AQHI: 已從 RSS 成功抓取 {len(v['aqhi'])} 個站點。")
    except Exception as e:
        print(f"❌ AQHI 抓取失敗: {e}")

    # 2. Wind CSV
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind_uc.csv")
        lines = r.text.strip().split('\n')[1:]
        wind_count = 0
        for line in lines:
            c = [x.strip('"').strip() for x in line.split(',')]
            for name, sid in STATION_MAP.items():
                if name in c[1]:
                    deg = wind_text_to_degrees(c[2])
                    try: spd = float(c[3])
                    except: spd = None
                    if deg is not None: 
                        fetched[f"PDIR_{sid}"] = deg
                        v["pdir"].append(deg)
                    if spd is not None: 
                        fetched[f"WSPD_{sid}"] = spd
                        v["wspd"].append(spd)
                        wind_count += 1
        print(f"✅ 風力: 已匹配 {wind_count} 個氣象站點數據。")
    except Exception as e:
        print(f"❌ 風力 CSV 解析失敗: {e}")

    # 3. Humidity JSON
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc")
        hum_data = r.json().get('humidity', {}).get('data', [])
        hum_count = 0
        for item in hum_data:
            for name, sid in STATION_MAP.items():
                if name in item['place']:
                    val = float(item['value'])
                    fetched[f"HUM_{sid}"] = val
                    v["hum"].append(val)
                    hum_count += 1
        print(f"✅ 濕度: 已匹配 {hum_count} 個濕度監測點數據。")
    except Exception as e:
        print(f"❌ 濕度 JSON 解析失敗: {e}")

    # 計算預設平均值
    means = {
        "AQHI": sum(v["aqhi"])/len(v["aqhi"]) if v["aqhi"] else 3.0,
        "HUM": sum(v["hum"])/len(v["hum"]) if v["hum"] else 80.0,
        "WSPD": sum(v["wspd"])/len(v["wspd"]) if v["wspd"] else 5.0,
        "PDIR": sum(v["pdir"])/len(v["pdir"]) if v["pdir"] else 180.0
    }
    
    print(f"💡 當前全港平均值參考: AQHI={round(means['AQHI'],1)}, HUM={round(means['HUM'],1)}%, 風速={round(means['WSPD'],1)}")
    return fetched, means

# --- [3. 執行同步與 CSV 寫入] ---
def run_sync():
    now_hkt = datetime.now(HKT)
    fetched, means = fetch_with_debug()

    # Firebase 更新
    try:
        if not firebase_admin._apps:
            creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_env)), {'databaseURL': FIREBASE_URL})
        
        fb_readings = {}
        for s in STATIONS_FIREBASE:
            clean_name = s.replace('_General','').replace('_Roadside','').replace('_','/')
            val = fetched.get(f"AQHI_{clean_name}", int(round(means["AQHI"])))
            fb_readings[s] = val
        
        db.reference("GAGNN_24hours/GAGNN_data").update({
            "last_updated": now_hkt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "readings": fb_readings
        })
        print("✅ Firebase 實時數據更新完成。")
    except Exception as e:
        print(f"⚠️ Firebase 更新警告: {e}")

    # CSV 寫入 (95 欄位) 與 數據來源檢查
    row = []
    real_count = 0
    default_count = 0

    for col in ALL_COLUMNS:
        if col == "Date":
            row.append(now_hkt.strftime("%Y-%m-%d"))
        elif col == "Cyclone_Present":
            row.append(0)
        elif col in fetched:
            row.append(fetched[col])
            real_count += 1
        else:
            # 填入預設值並記錄
            default_count += 1
            if "AQHI" in col: row.append(round(means["AQHI"]))
            elif "HUM" in col: row.append(round(means["HUM"], 1))
            elif "WSPD" in col: row.append(round(means["WSPD"], 1))
            elif "PDIR" in col: row.append(round(means["PDIR"], 1))
            else: row.append(0.0)

    # 打印數據健康度報告
    total_metrics = real_count + default_count
    health_score = (real_count / total_metrics) * 100
    print(f"📊 數據存檔報告: [真實值: {real_count}] [預設值: {default_count}] [健康度: {health_score:.1f}%]")
    if health_score < 50:
        print("🚩 警告: 超過一半的數據使用了預設值，請檢查 API 站點映射。")

    # 儲存
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(map(str, row)) + "\n")
    
    print(f"🏁 同步結束時間: {now_hkt.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_sync()
