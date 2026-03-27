import firebase_admin
from firebase_admin import credentials, db
import requests
import re
from datetime import datetime
import os
import json

FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def fetch_aqhi_by_regex():
    """使用正则表达式直接从 JS 内容中提取数据"""
    api_url = "https://www.aqhi.gov.hk/js/data/aqhi_data.js"
    try:
        res = requests.get(api_url, timeout=15)
        content = res.text
        
        # 匹配模式：寻找 "station_en":"站名" 和 "aqhi":"数字"
        # 正则逻辑：抓取 station_en 后的字符串和 aqhi 后的数字
        stations = re.findall(r'"station_en":"([^"]+)"', content)
        values = re.findall(r'"aqhi":"([^"]+)"', content)
        
        if not stations or not values:
            print("⚠️ 正则匹配未找到数据，尝试备用匹配模式...")
            # 备用：匹配非引号包围的数值
            values = re.findall(r'"aqhi":([\d\+]+)', content)

        aqhi_dict = {}
        # 配对提取
        for s, v in zip(stations, values):
            # 过滤掉非数字的 AQHI (比如 'N/A')
            if v.isdigit() or (v.endswith('+') and v[:-1].isdigit()):
                clean_name = s.replace('/', '_').replace(' ', '_')
                aqhi_dict[clean_name] = v
        
        return aqhi_dict
    except Exception as e:
        print(f"❌ 正则提取失败: {e}")
        return None

def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json:
        print("❌ 找不到环境变量")
        return

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        # 使用正则方式获取数据
        aqhi_results = fetch_aqhi_by_regex()
        
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results and len(aqhi_results) > 0:
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "aqhi_readings": aqhi_results,
                "status": "online",
                "method": "Regex_Extraction"
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 正则匹配成功！同步了 {len(aqhi_results)} 个站点。")
            print(f"预览: {list(aqhi_results.items())[:5]}")
        else:
            print("⚠️ 匹配结果为空。")
            ref.update({"status": "regex_empty", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    except Exception as e:
        print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    run_sync()
