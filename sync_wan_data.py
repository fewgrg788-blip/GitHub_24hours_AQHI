import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import re
import os
import pandas as pd
import json
from datetime import datetime, timedelta, timezone

# --- 配置 ---
RENDER_GNN_API = os.getenv("RENDER_GNN_API", "https://your-gagnn-service.onrender.com/predict")
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"
HKT = timezone(timedelta(hours=8))

STATIONS = [
    "Central_Western_General", "Eastern_General", "Kwai_Chung_General", "Kwun_Tong_General",
    "North_General", "Sha_Tin_General", "Sham_Shui_Po_General", "Southern_General",
    "Tai_Po_General", "Tap_Mun_General", "Tseung_Kwan_O_General", "Tsuen_Wan_General",
    "Tuen_Mun_General", "Tung_Chung_General", "Yuen_Long_General",
    "Causeway_Bay_Roadside", "Central_Roadside", "Mong_Kok_Roadside"
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
            m = re.search(r'(\d{1,2}\+?)', re.sub(r'\d{4}', '', desc))
            if m:
                val = m.group(1).replace('+', '')
                key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
                key += "_Roadside" if "Roadside" in desc else "_General"
                results[key] = float(val)
        print(f"✅ Successfully fetched data for {len(results)} stations.")
        return results
    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        return None

def get_gnn_prediction(current_readings):
    print(f"🧠 [Step 4] Requesting GNN prediction from Render: {RENDER_GNN_API}")
    try:
        input_vector = [current_readings.get(s, 0) for s in STATIONS]
        response = requests.post(RENDER_GNN_API, json={"data": input_vector}, timeout=45)
        
        if response.status_code == 200:
            pred_data = response.json().get("prediction")
            print("✅ GNN Prediction received successfully.")
            return {STATIONS[i]: pred_data[i] for i in range(len(STATIONS))}
        else:
            print(f"⚠️ Render returned status code: {response.status_code}")
    except Exception as e:
        print(f"⚠️ GNN Prediction Error (Render might be sleeping/offline): {e}")
    return None

def run_integration():
    now_hkt = datetime.now(HKT)
    current_time_str = now_hkt.strftime("%Y-%m-%d %H:00")
    target_time_str = (now_hkt + timedelta(hours=6)).strftime("%Y-%m-%d %H:00")

    print(f"\n🚀 === BuildTech Sync Start: {current_time_str} HKT ===")
    
    # 1. Firebase 初始化
    try:
        if not firebase_admin._apps:
            print("🔑 Initializing Firebase Admin SDK...")
            creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            if not creds_env:
                raise ValueError("FIREBASE_SERVICE_ACCOUNT Secret is missing!")
            creds = json.loads(creds_env)
            firebase_admin.initialize_app(credentials.Certificate(creds), {'databaseURL': FIREBASE_URL})
    except Exception as e:
        print(f"❌ Firebase Init Error: {e}"); return

    # 2. 獲取實時數據
    actual_data = fetch_aqhi()
    if not actual_data: return

    # 3. 核心驗證：比對 6 小時前存下的預測
    print(f"📊 [Step 3] Checking for past prediction made for {current_time_str}...")
    history_ref = db.reference(f"GAGNN_v2/predictions/{current_time_str}")
    predicted_6h_ago = history_ref.get()

    avg_error = 0
    credibility_msg = "Initial Sync (No past data)"
    if predicted_6h_ago:
        errors = [abs(actual_data.get(s, 0) - predicted_6h_ago.get(s, 0)) for s in STATIONS]
        avg_error = sum(errors) / len(STATIONS)
        credibility_msg = f"Verified (MAE: {avg_error:.2f})"
        print(f"✅ Comparison Complete. MAE: {avg_error:.2f}")
    else:
        print("ℹ️ No past prediction found for this hour. Skipping MAE calculation.")

    # 4. 獲取未來 6 小時預測
    new_prediction = get_gnn_prediction(actual_data)
    if new_prediction:
        db.reference(f"GAGNN_v2/predictions/{target_time_str}").set(new_prediction)
        print(f"🔮 Prediction for {target_time_str} saved to Firebase.")

    # 5. 更新 Dashboard 面板
    accuracy_score = round(max(0, 100 - avg_error * 10), 2) if predicted_6h_ago else 100
    db.reference("GAGNN_v2/dashboard").set({
        "last_updated": current_time_str,
        "current_avg": round(sum(actual_data.values()) / len(STATIONS), 2),
        "accuracy_score": accuracy_score,
        "verification_status": credibility_msg,
        "prediction_target": target_time_str
    })
    print(f"📱 Dashboard updated with Score: {accuracy_score}")

    # 6. 更新 CSV 歷史紀錄
    print(f"💾 [Step 6] Appending data to {CSV_FILE}...")
    row = {
        "timestamp": current_time_str,
        "avg_actual": round(sum(actual_data.values()) / len(STATIONS), 2),
        "mae_error": round(avg_error, 3) if predicted_6h_ago else 0
    }
    for s in STATIONS:
        row[f"{s}_actual"] = actual_data.get(s, 0)
        row[f"{s}_pred_6h_ago"] = predicted_6h_ago.get(s, "N/A") if predicted_6h_ago else "N/A"

    df_new = pd.DataFrame([row])
    if not os.path.isfile(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
    else:
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
    
    print(f"🏁 === BuildTech Sync Finished ===\n")

if __name__ == "__main__":
    run_integration()
