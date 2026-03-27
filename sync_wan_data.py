import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import re
import os
import pandas as pd
from datetime import datetime
import json
from datetime import datetime, timedelta, timezone

# 强制使用 HKT (UTC+8)
hkt = timezone(timedelta(hours=8))
timestamp = datetime.now(hkt).strftime("%Y-%m-%d %H:%M")

# --- 配置 ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"

# 固定站点列表，确保 CSV 列顺序和 GNN 节点索引一致
STATIONS = [
    "Central_Western_General", "Eastern_General", "Kwai_Chung_General", "Kwun_Tong_General",
    "North_General", "Sha_Tin_General", "Sham_Shui_Po_General", "Southern_General",
    "Tai_Po_General", "Tap_Mun_General", "Tseung_Kwan_O_General", "Tsuen_Wan_General",
    "Tuen_Mun_General", "Tung_Chung_General", "Yuen_Long_General",
    "Causeway_Bay_Roadside", "Central_Roadside", "Mong_Kok_Roadside"
]

def fetch_aqhi():
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    try:
        res = requests.get(api_url, timeout=15)
        root = ET.fromstring(res.content.decode('utf-8', errors='ignore'))
        results = {}
        for item in root.findall(".//item"):
            title = item.find("title").text
            desc = item.find("description").text
            m = re.search(r'Stations:\s*(\d{1,2}\+?)', desc)
            if not m:
                m = re.search(r'(\d{1,2}\+?)', re.sub(r'\d{4}', '', desc))
            
            if m:
                val = m.group(1)
                key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
                key += "_Roadside" if "Roadside" in desc else "_General"
                results[key] = val
        return results
    except Exception as e:
        print(f"Fetch Error: {e}")
        return None

def run_integration():
    print(f"--- 启动任务: {datetime.now()} ---")
    data = fetch_aqhi()
    if not data: return

    # 1. Firebase 集成
    creds = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(creds), {'databaseURL': FIREBASE_URL})
    
    # 更新实时节点
    db.reference("GAGNN_24hours/GAGNN_data").set({"last_updated": str(datetime.now()), "readings": data})
    
    # 2. CSV 集成 (自动化累积训练数据)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = {"timestamp": timestamp}
    # 按照固定顺序填入数据，缺失补 0 或 N/A
    for s in STATIONS:
        row[s] = data.get(s, "0") 

    df_new = pd.DataFrame([row])
    if not os.path.isfile(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
    else:
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
    
    print(f"✅ Firebase & CSV 同步完成: {timestamp}")

if __name__ == "__main__":
    run_integration()
