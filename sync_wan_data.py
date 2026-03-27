import firebase_admin
from firebase_admin import credentials, db
import requests
from datetime import datetime
import os
import json

FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def fetch_hk_aqhi_api():
    """直接从环保署的 JSON 接口获取数据，跳过 HTML 解析"""
    # 这是环保署官网 JS 调用的真实数据接口
    api_url = "https://www.aqhi.gov.hk/js/data/aqhi_data.js"
    try:
        res = requests.get(api_url, timeout=15)
        # 该接口返回的是 JS 变量赋值语句，我们需要提取里面的 JSON 部分
        # 格式通常是: var aqhi_data = [...];
        content = res.text
        json_str = content[content.find('['):content.rfind(']')+1]
        data_list = json.loads(json_str)
        
        aqhi_dict = {}
        for item in data_list:
            # 接口字段映射：station_en 是站名，aqhi 是数值
            station = item.get('station_en', '').replace('/', '_').replace(' ', '_')
            value = item.get('aqhi', 'N/A')
            if station and value != 'N/A':
                aqhi_dict[station] = value
                
        return aqhi_dict
    except Exception as e:
        print(f"❌ API 抓取失败: {e}")
        return None

def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json: return

    try:
        # 1. 初始化 Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        # 2. 获取数据 (改用 API 方式)
        aqhi_results = fetch_hk_aqhi_api()
        
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results:
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "aqhi_readings": aqhi_results,
                "status": "online",
                "method": "API_JS_Source"
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 成功从 API 同步了 {len(aqhi_results)} 个站点数据！")
        else:
            print("⚠️ 未能获取到数据。")
            ref.update({"status": "api_fetch_failed"})

    except Exception as e:
        print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    run_sync()
