import firebase_admin
from firebase_admin import credentials, db
import requests
from datetime import datetime
import os
import json

# --- 配置 ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def fetch_official_aqhi_api():
    """从 data.gov.hk 官方接口获取实时数据"""
    # 官方 JSON 数据源 URL
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_en.json"
    try:
        print(f"正在调用官方 API: {api_url}")
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        
        # 官方返回的格式通常是: [{"station": "Central/Western", "aqhi": "4", ...}, ...]
        raw_data = res.json()
        
        aqhi_dict = {}
        for item in raw_data:
            # 提取站名并清洗 (Firebase 键名不支持 /)
            station = item.get('station', 'Unknown').replace('/', '_').replace(' ', '_')
            aqhi_val = item.get('aqhi', 'N/A')
            
            # 记录数据
            if station != 'Unknown' and aqhi_val != 'N/A':
                aqhi_dict[station] = aqhi_val
                
        return aqhi_dict
    except Exception as e:
        print(f"❌ 官方 API 调用失败: {e}")
        return None

def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    
    # 1. 检查 Secrets
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json:
        print("❌ 错误: 找不到 FIREBASE_SERVICE_ACCOUNT Secret")
        return

    try:
        # 2. 初始化 Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        # 3. 获取官方数据
        aqhi_results = fetch_official_aqhi_api()
        
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results and len(aqhi_results) > 0:
            # 4. 写入 Firebase
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "aqhi_readings": aqhi_results,
                "status": "online",
                "source": "data.gov.hk_official"
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 官方 API 同步成功！共 {len(aqhi_results)} 个站点。")
            print(f"实时数据预览: {list(aqhi_results.items())[:3]}")
        else:
            print("⚠️ 接口返回数据为空。")
            ref.update({"status": "api_empty", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    except Exception as e:
        print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    run_sync()
