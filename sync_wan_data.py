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
    """从官方 XML 获取 AQHI 数据（改进版）"""
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    
    try:
        print(f"正在调用官方 XML: {api_url}")
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        
        root = ET.fromstring(res.content)
        aqhi_dict = {}
        
        # 提取更新时间
        pub_date = root.find(".//pubDate")
        source_time = pub_date.text if pub_date is not None else None
        
        # 遍历所有 item
        for item in root.findall(".//item"):
            title = item.find("title")
            description = item.find("description")
            
            if title is not None and description is not None:
                station = title.text.strip() if title.text else ""
                desc_text = description.text.strip() if description.text else ""
                
                # 从 description 中提取 AQHI 数值（支持 1-10 或 10+）
                aqhi_match = re.search(r':\s*(\d+\+?)\s', desc_text)
                if aqhi_match:
                    aqhi_val = aqhi_match.group(1)
                    
                    # 清洗站名（适合 Firebase Key）
                    clean_station = station.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
                    
                    # 区分 General / Roadside（可选更精确）
                    if "Roadside" in desc_text:
                        clean_station += "_Roadside"
                    elif "General" in desc_text:
                        clean_station += "_General"
                    
                    aqhi_dict[clean_station] = aqhi_val
        
        print(f"成功解析 {len(aqhi_dict)} 个监测站")
        return aqhi_dict, source_time
        
    except Exception as e:
        print(f"❌ 官方 XML 调用失败: {e}")
        return None, None


def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json:
        print("❌ 错误: 找不到 FIREBASE_SERVICE_ACCOUNT Secret")
        return
    
    try:
        # 初始化 Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        # 获取数据
        aqhi_results, source_time = fetch_official_aqhi_xml()
        
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results and len(aqhi_results) >= 10:   # 正常应有 15-18 个左右
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_last_updated": source_time,
                "aqhi_readings": aqhi_results,
                "status": "online",
                "method": "Official_XML_Eng_Improved"
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 同步成功！共 {len(aqhi_results)} 个站点。")
            print(f"数据预览: {list(aqhi_results.items())[:6]}")
        else:
            print(f"⚠️ 数据不足，仅解析到 {len(aqhi_results) if aqhi_results else 0} 个站点。")
            ref.update({"status": "xml_insufficient", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            
    except Exception as e:
        print(f"❌ 运行异常: {e}")


if __name__ == "__main__":
    run_sync()
