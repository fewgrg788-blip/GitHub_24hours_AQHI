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
# 建議將 Render API URL 加入 GitHub Secrets
RENDER_GNN_API = "https://your-gagnn-service.onrender.com/predict" 
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
CSV_FILE = "aqhi_history.csv"

# 强制使用 HKT (UTC+8)
HKT = timezone(timedelta(hours=8))

STATIONS = [
    "Central_Western_General", "Eastern_General", "Kwai_Chung_General", "Kwun_Tong_General",
    "North_General", "Sha_Tin_General", "Sham_Shui_Po_General", "Southern_General",
    "Tai_Po_General", "Tap_Mun_General", "Tseung_Kwan_O_General", "Tsuen_Wan_General",
    "Tuen_Mun_General", "Tung_Chung_General", "Yuen_Long_General",
    "Causeway_Bay_Roadside", "Central_Roadside", "Mong_Kok_Roadside"
]

def fetch_aqhi():
    """從政府 RSS 抓取實時 AQHI"""
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
                val = m.group(1).replace('+', '') # 移除 10+ 的加號方便運算
                key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
                key += "_Roadside" if "Roadside" in desc else "_General"
                results[key] = float(val)
        return results
    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        return None

def get_gnn_prediction(current_readings):
    """將當前數據發送到 Render 上的 GAGNN 模型進行 6 小時預測"""
    try:
        # 按照 STATIONS 順序準備輸入向量
        input_vector = [current_readings.get(s, 0) for s in STATIONS]
        
        # 喚醒並請求 Render (建議 Render 端接收 JSON 並回傳 18 區的預測陣列)
        response = requests.post(RENDER_GNN_API, json={"data": input_vector}, timeout=30)
        if response.status_code == 200:
            pred_data = response.json().get("prediction") # 假設回傳格式為 {"prediction": [...]}
            # 將預測結果轉回字典格式
            return {STATIONS[i]: pred_data[i] for i in range(len(STATIONS))}
    except Exception as e:
        print(f"⚠️ GNN Prediction Error (Render might be sleeping): {e}")
    return None

def run_integration():
    now_hkt = datetime.now(HKT)
    current_time_str = now_hkt.strftime("%Y-%m-%d %H:00") # 歸一化到整點
    target_time_str = (now_hkt + timedelta(hours=6)).strftime("%Y-%m-%d %H:00")
    past_time_str = (now_hkt - timedelta(hours=6)).strftime("%Y-%m-%d %H:00")

    print(f"--- 啟動任務: {current_time_str} HKT ---")
    
    # 1. 初始化 Firebase
    if not firebase_admin._apps:
        creds = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
        firebase_admin.initialize_app(credentials.Certificate(creds), {'databaseURL': FIREBASE_URL})

    # 2. 抓取實時數據 (Actual)
    actual_data = fetch_aqhi()
    if not actual_data: return

    # 3. 核心：驗證 6 小時前的預測是否準確
    # 從 Firebase 讀取 6 小時前為「現在」做的預測
    history_ref = db.reference(f"GAGNN_v2/predictions/{current_time_str}")
    predicted_data_6h_ago = history_ref.get()

    avg_error = 0
    if predicted_data_6h_ago:
        # 計算平均絕對誤差 (MAE)
        errors = [abs(actual_data.get(s, 0) - predicted_data_6h_ago.get(s, 0)) for s in STATIONS]
        avg_error = sum(errors) / len(STATIONS)
        print(f"📊 驗證完成: 6小時前預測誤差 (MAE) = {avg_error:.2f}")

    # 4. 獲取新的預測 (Future Simulation for T+6)
    new_prediction = get_gnn_prediction(actual_data)
    if new_prediction:
        db.reference(f"GAGNN_v2/predictions/{target_time_str}").set(new_prediction)
        print(f"🔮 已儲存對 {target_time_str} 的模擬結果")

    # 5. 更新 Firebase 實時面板 (供 App/Web 顯示)
    db.reference("GAGNN_v2/dashboard").set({
        "last_updated": current_time_str,
        "current_readings": actual_data,
        "accuracy_score": round(max(0, 100 - avg_error * 10), 2), # 簡單轉化為百分比信用分
        "prediction_6h_target": target_time_str
    })

    # 6. CSV 數據累加 (用於長期公信力證明)
    row = {
        "timestamp": current_time_str,
        "avg_actual": sum(actual_data.values()) / len(actual_data),
        "mae_error": avg_error
    }
    # 加入各站點數據
    for s in STATIONS:
        row[f"{s}_actual"] = actual_data.get(s, 0)
        row[f"{s}_pred_6h_ago"] = predicted_data_6h_ago.get(s, "N/A") if predicted_data_6h_ago else "N/A"

    df_new = pd.DataFrame([row])
    if not os.path.isfile(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
    else:
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
    
    print(f"✅ 同步完成。公信力分數: {row.get('accuracy_score', 'N/A')}")

if __name__ == "__main__":
    run_integration()
