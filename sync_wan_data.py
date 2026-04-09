import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import re
import os
import json
import urllib.parse
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

# --- [2. 數據獲取與核心修復函數] ---
def fetch_aqhi():
    print("🌐 [Step 1] Fetching real-time AQHI from EPD RSS...")
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    try:
        res = requests.get(api_url, timeout=15)
        root = ET.fromstring(res.content.decode('utf-8', errors='ignore'))
        results = {}
        raw_data = {} # 暫存原始數據進行檢查

        for item in root.findall(".//item"):
            desc = item.find("description").text
            title = item.find("title").text
            m = re.search(r'(\d{1,2})', re.sub(r'\d{4}', '', desc))
            key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
            key += "_Roadside" if "Roadside" in desc else "_General"

            if m:
                val = int(m.group(1))
                if val > 0 and val <= 11: # 基本過濾
                    raw_data[key] = val
                else:
                    raw_data[key] = None
            else:
                raw_data[key] = None

        # --- [新增：異常值檢測邏輯] ---
        # 1. 先算出目前所有有效數字的初步平均值
        temp_list = [v for v in raw_data.values() if v is not None]
        pre_avg = sum(temp_list) / len(temp_list) if temp_list else 3
        
        # 2. 進行合理性檢查：如果某站點比平均值高出 4 級以上，視為錯誤 (如平均 3, 它卻 9)
        final_valid_values = []
        for s in STATIONS:
            val = raw_data.get(s)
            # 如果數值存在，且與全港平均差距在合理範圍內 (比如 <= 4)
            if val is not None and abs(val - pre_avg) <= 4:
                results[s] = val
                final_valid_values.append(val)
            else:
                # 如果是那個異常的 9，這裡會將其設為 None
                print(f"🚨 [Outlier Detection] Station {s} has abnormal value ({val}). Marking for repair.")
                results[s] = None

        # 3. 均值填充 (Imputation)
        global_avg_int = int(round(sum(final_valid_values) / len(final_valid_values))) if final_valid_values else 3
        for s in STATIONS:
            if results.get(s) is None:
                print(f"🩹 [Repair] Replacing error value in {s} with: {global_avg_int}")
                results[s] = global_avg_int

        return results
    except Exception as e:
        print(f"❌ Fetch Error: {e}"); return None


# --- [3. AI 模型推理] ---
def get_gnn_prediction(current_readings):
    print(f"🧠 [Step 4] Requesting GNN prediction from Render...")
    try:
        # 確保輸入給模型的始終是有效的 18 個數值
        input_vector = [current_readings.get(s, 3) for s in STATIONS]
        response = requests.post(RENDER_GNN_API, json={"data": input_vector}, timeout=45)
        if response.status_code == 200:
            pred_data = response.json().get("prediction")
            print("✅ GNN Prediction successful.")
            return {STATIONS[i]: pred_data[i] for i in range(len(STATIONS))}
    except Exception as e:
        print(f"⚠️ GNN Service Error: {e}")
    return None

# --- [4. 主同步邏輯] ---
def run_integration():
    now_hkt = datetime.now(HKT)
    display_time = now_hkt.strftime("%Y-%m-%d %H:00")
    target_time_str = (now_hkt + timedelta(hours=6)).strftime("%Y-%m-%d %H:00")
    # 保持原有的詳細時間戳格式，確保不影響原有前端讀取
    timestamp_full = now_hkt.strftime("%Y-%m-%d %H:%M:%S.%f")

    print(f"\n🚀 === BuildTech Sync Start: {display_time} HKT ===")
    
    # Firebase 初始化
    try:
        if not firebase_admin._apps:
            creds_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_env)), {'databaseURL': FIREBASE_URL})
    except Exception as e:
        print(f"❌ Firebase Init Error: {e}"); return

    # 1. 抓取並自動修補數據
    actual_data = fetch_aqhi()
    if not actual_data: return
    current_avg = round(sum(actual_data.values()) / len(STATIONS), 2)

    # 2. 更新實時數據區 (GAGNN_24hours) - 保持格式，修正類型
    print(f"📡 [Log] Updating GAGNN_24hours/GAGNN_data...")
    db.reference("GAGNN_24hours/GAGNN_data").update({
        "last_updated": timestamp_full,
        "readings": actual_data
    })

    # 3. 處理 V2 驗證與預測區 (GAGNN_v2)
    # 使用 URL 編碼處理 Key 中的冒號與空格，解決 400 錯誤
    safe_current_path = urllib.parse.quote(display_time)
    safe_target_path = urllib.parse.quote(target_time_str)

    # 讀取 6 小時前的預測紀錄進行 MAE 對比
    past_pred = db.reference(f"GAGNN_v2/predictions/{safe_current_path}").get()

    avg_error = 0
    status_msg = "Initial Sync (No past data)"
    if past_pred:
        errs = [abs(actual_data.get(s, 0) - past_pred.get(s, 0)) for s in STATIONS]
        avg_error = sum(errs) / len(STATIONS)
        status_msg = f"Verified (MAE: {avg_error:.2f})"
        print(f"📊 [Log] Accuracy verified. MAE: {avg_error:.2f}")

    # 獲取新的預測
    new_prediction = get_gnn_prediction(actual_data)
    if new_prediction:
        db.reference(f"GAGNN_v2/predictions/{safe_target_path}").set(new_prediction)
        print(f"🔮 [Log] Prediction for {target_time_str} saved.")

    # 更新 Dashboard 面板
    db.reference("GAGNN_v2/dashboard").set({
        "last_updated": display_time,
        "current_avg": current_avg,
        "accuracy_score": round(max(0, 100 - avg_error * 10), 2) if past_pred else 100,
        "verification_status": status_msg,
        "prediction_target": target_time_str
    })

    # 4. 寫入 CSV 歷史紀錄
    # 確保寫入 CSV 的也是乾淨的整數等級
    levels = [str(actual_data.get(s)) for s in STATIONS]
    csv_row = f"{display_time},{current_avg}," + ",".join(levels)

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            short_names = [s.replace('_General','').replace('_Roadside','R') for s in STATIONS]
            f.write("Time,Avg," + ",".join(short_names) + "\n")
        f.write(csv_row + "\n")
    
    print(f"🏁 === BuildTech Sync Finished ===\n")

if __name__ == "__main__":
    run_integration()
