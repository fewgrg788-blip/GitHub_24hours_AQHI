import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
import re
from datetime import datetime
import os
import json

# --- 配置 ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def fetch_official_aqhi_xml():
    """最终修复版 - 严格匹配官方真实 XML 格式"""
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    
    try:
        print(f"正在调用官方 XML: {api_url}")
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        
        root = ET.fromstring(res.content)
        aqhi_dict = {}
        
        pub_date = root.find(".//pubDate")
        source_time = pub_date.text if pub_date is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        success_count = 0
        for item in root.findall(".//item"):
            title = item.find("title")
            description = item.find("description")
            
            if not title or not description or not title.text or not description.text:
                continue
                
            station = title.text.strip()
            desc = description.text.strip()
            
            # 关键修复：更宽松、准确的正则（匹配 ": 4 Moderate" 或 ": 5 " 等格式）
            match = re.search(r':\s*(\d{1,2}\+?)\s', desc)
            if match:
                aqhi_val = match.group(1)
                success_count += 1
                
                # 清洗站名
                clean_station = re.sub(r'[^a-zA-Z0-9_]', '_', station)
                clean_station = re.sub(r'_+', '_', clean_station).strip('_')
                
                if "Roadside" in desc:
                    clean_station += "_Roadside"
                else:
                    clean_station += "_General"
                
                aqhi_dict[clean_station] = aqhi_val
        
        print(f"✅ 成功解析 {len(aqhi_dict)} 个监测站（匹配成功 {success_count} 条）")
        if aqhi_dict:
            print("预览前 6 个站点:", dict(list(aqhi_dict.items())[:6]))
        
        return aqhi_dict, source_time
        
    except Exception as e:
        print(f"❌ XML 获取失败: {e}")
        return None, None


def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json:
        print("❌ 错误: 找不到 FIREBASE_SERVICE_ACCOUNT Secret")
        return
    
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        aqhi_results, source_time = fetch_official_aqhi_xml()
        
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results and len(aqhi_results) >= 10:
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_last_updated": source_time,
                "aqhi_readings": aqhi_results,
                "status": "online",
                "method": "Official_XML_Eng_Final_v3"
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 同步成功！共 {len(aqhi_results)} 个站点。")
        else:
            print(f"⚠️ 数据不足，仅获取到 {len(aqhi_results) if aqhi_results else 0} 个站点")
            ref.update({
                "status": "data_insufficient", 
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stations_count": len(aqhi_results) if aqhi_results else 0,
                "last_attempt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
    except Exception as e:
        print(f"❌ 运行异常: {e}")


if __name__ == "__main__":
    run_sync()
