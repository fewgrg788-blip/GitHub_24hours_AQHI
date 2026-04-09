import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import re
import os
import json
import urllib.parse  # 引入 URL 編碼工具
from datetime import datetime, timedelta, timezone

# --- [1. 配置] ---
RENDER_GNN_API = "https://buildtech-gnn-service.onrender.com/predict"
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

def fetch_aqhi():
    print("🌐 [Step 1] Fetching real-time AQHI from EPD RSS...")
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    try:
        res = requests.get(api_url, timeout=15)
        root = ET.fromstring(res.content.decode('utf-8', errors='ignore'))
        results = {}
        for item in root.findall(".//item"):
            desc = item.find("description").text
            title = item.find("title").text
            m = re.search(r'(\d{1,2})', re.sub(r'\d{4}', '', desc))
            if m:
                val = float(m.group(1))
                val = max(1.0, min(11.0, val))
                key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
                key += "_Roadside" if "Roadside" in desc else "_General"
                results[key] = val
        return results
    except Exception as e:
        print(f"❌ Fetch Error: {e}"); return None

def get_gnn_prediction(current_readings):
    print(f"🧠 [Step 4] Requesting GNN prediction from Render...")
    try:
        input_vector = [current_readings.get(s, 0) for s in STATIONS]
        response = requests.post(RENDER_GNN_API, json={"data": input_vector}, timeout=45)
        if response.status_code == 200:
            pred_data = response.json().get("prediction")
            return {STATIONS[i]: pred_data[i] for i in range(len(STATIONS))}
    except Exception as e:
        print(f"⚠️ GNN Offline: {e}")
    return None

def run_integration():
    now_hkt = datetime.now(HKT)
    # 你想要的標準格式
    current_time_str = now_hkt.strftime("%Y-%m-%d %H:00")
    target_time_str = (now_hkt + timedelta(hours=6)).strftime("%Y-%m-%d %H:00")

    print(f"\n🚀 === BuildTech Sync Start: {current_time_str} HKT ===")
    
    # Firebase 初始化
    try:
        if not firebase_admin._apps:
            creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_env)), {'databaseURL': FIREBASE_URL})
    except Exception as e:
        print(f"❌ Firebase Error: {e}"); return

    actual_data = fetch_aqhi()
    if not actual_data: return
    current_avg = round(sum(actual_data.values()) / len(STATIONS), 2)

    # --- [關鍵修復：URL 安全編碼] ---
    # 使用 quote 將 "2026-04-09 22:00" 轉為 "2026-04-09%2022%3A00"
    safe_current_path = urllib.parse.quote(current_time_str)
    safe_target_path = urllib.parse.quote(target_time_str)

    print(f"📊 [Step 3] Checking for past prediction for {current_time_str}...")
    past_pred_ref = db.reference(f"GAGNN_v2/predictions/{safe_current_path}")
    predicted_6h_ago = past_pred_ref.get()

    avg_error = 0
    status_msg = "Initial Sync"
    if predicted_6h_ago:
        errs = [abs(actual_data.get(s, 0) - predicted_6h_ago.get(s, 0)) for s in STATIONS]
        avg_error = sum(errs) / len(STATIONS)
        status_msg = f"Verified (MAE: {avg_error:.2f})"

    new_prediction = get_gnn_prediction(actual_data)
    if new_prediction:
        # 使用轉義後的路徑寫入，但在 Firebase 顯示會是原始字串
        db.reference(f"GAGNN_v2/predictions/{safe_target_path}").set(new_prediction)
        print(f"🔮 Prediction for {target_time_str} saved successfully.")

    accuracy_score = round(max(0, 100 - avg_error * 10), 2) if predicted_6h_ago else 100
    db.reference("GAGNN_v2/dashboard").set({
        "last_updated": current_time_str,
        "current_avg": current_avg,
        "accuracy_score": accuracy_score,
        "verification_status": status_msg,
        "prediction_target": target_time_str
    })

    # --- [Step 6: CSV 儲存] ---
    levels = [str(int(round(actual_data.get(s, 0)))) for s in STATIONS]
    csv_row = f"{current_time_str},{current_avg}," + ",".join(levels)

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            short_names = [s.replace('_General','').replace('_Roadside','R') for s in STATIONS]
            f.write("Time,Avg," + ",".join(short_names) + "\n")
        f.write(csv_row + "\n")
    
    print(f"🏁 === BuildTech Sync Finished ===\n")

if __name__ == "__main__":
    run_integration()
