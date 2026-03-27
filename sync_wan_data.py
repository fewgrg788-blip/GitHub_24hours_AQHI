import firebase_admin
from firebase_admin import credentials, db
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import os
import json

# --- 配置 ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def fetch_official_aqhi_xml():
    """从官方 XML 获取所有监测站的实时 AQHI（推荐方式）"""
    # 英文版 XML（推荐）
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    
    try:
        print(f"正在调用官方 XML: {api_url}")
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        
        root = ET.fromstring(res.content)
        aqhi_dict = {}
        
        # 提取最后更新时间
        last_updated = None
        pub_date = root.find(".//pubDate")
        if pub_date is not None:
            last_updated = pub_date.text
        
        # 遍历所有 <item>，提取站名和 AQHI
        for item in root.findall(".//item"):
            title = item.find("title")
            if title is not None and title.text:
                title_text = title.text.strip()
                
                # 示例 title: "Central/Western: 4" 或 "Mong Kok (Roadside): 5"
                if ":" in title_text:
                    station_part, aqhi_part = title_text.split(":", 1)
                    station = station_part.strip()
                    aqhi_val = aqhi_part.strip()
                    
                    # 清洗站名（Firebase Key 不能有 / 和空格）
                    clean_station = station.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                    
                    # 只保留数字或带+的数值
                    if aqhi_val.replace('+', '').isdigit():
                        aqhi_dict[clean_station] = aqhi_val
        
        print(f"成功解析 {len(aqhi_dict)} 个监测站")
        return aqhi_dict, last_updated
        
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
        
        # 获取官方数据
        aqhi_results, source_time = fetch_official_aqhi_xml()
        
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results and len(aqhi_results) > 5:   # 正常应该有 15+ 个站点
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_last_updated": source_time,
                "aqhi_readings": aqhi_results,
                "status": "online",
                "method": "Official_XML_Eng"
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 官方 XML 同步成功！共 {len(aqhi_results)} 个站点。")
            print(f"数据预览: {list(aqhi_results.items())[:5]}")
        else:
            print("⚠️ 接口返回数据不足或为空。")
            ref.update({"status": "xml_empty", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            
    except Exception as e:
        print(f"❌ 运行异常: {e}")


if __name__ == "__main__":
    run_sync()
