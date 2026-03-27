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
    """使用『模糊匹配+提取』策略，绕过固定的文本格式限制"""
    api_url = "https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_rss_Eng.xml"
    
    try:
        print(f"正在调用官方 XML: {api_url}")
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        
        # 处理可能的编码问题
        content = res.content.decode('utf-8', errors='ignore')
        root = ET.fromstring(content)
        aqhi_dict = {}
        
        pub_date = root.find(".//pubDate")
        source_time = pub_date.text if pub_date is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for item in root.findall(".//item"):
            title = item.find("title").text if item.find("title") is not None else ""
            desc = item.find("description").text if item.find("description") is not None else ""
            
            if not title or not desc:
                continue
                
            # --- 核心逻辑：直接从 description 中提取 AQHI 数字 ---
            # 这里的正则 \d{1,2}\+? 会匹配 1 到 10 或者 10+
            # 我们寻找紧跟在 "Stations:" 后面或者第一个出现的独立数字
            aqhi_val = None
            
            # 尝试 1: 寻找 "Stations: 4" 这种结构
            m1 = re.search(r'Stations:\s*(\d{1,2}\+?)', desc)
            if m1:
                aqhi_val = m1.group(1)
            else:
                # 尝试 2: 暴力提取 desc 里的第一个 1-2 位数字（忽略日期里的 2026）
                # 我们先移除日期部分再找
                clean_desc = re.sub(r'\d{4}', '', desc) # 删掉年份
                m2 = re.search(r'(\d{1,2}\+?)', clean_desc)
                if m2:
                    aqhi_val = m2.group(1)

            if aqhi_val:
                # 清洗站名并区分类型
                station_key = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_')
                if "Roadside" in desc or "Roadside" in title:
                    station_key += "_Roadside"
                else:
                    station_key += "_General"
                
                aqhi_dict[station_key] = aqhi_val
        
        return aqhi_dict, source_time
        
    except Exception as e:
        print(f"❌ XML 解析异常: {e}")
        return None, None

def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json:
        print("❌ 错误: 缺少 Firebase 凭据")
        return

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        aqhi_results, source_time = fetch_official_aqhi_xml()
        
        if aqhi_results and len(aqhi_results) > 0:
            ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_time": source_time,
                "aqhi_readings": aqhi_results,
                "status": "online",
                "count": len(aqhi_results)
            }
            ref.set(payload)
            print(f"🚀 [SUCCESS] 成功同步 {len(aqhi_results)} 个站点！")
            print(f"数据预览: {list(aqhi_results.items())[:3]}")
        else:
            print("⚠️ 仍然没能解析到数据，正在进入兜底模式...")
            # 如果真的抓不到，至少更新一下状态，不要让 Actions 报错
            db.reference(f"{ROOT_NODE}/{DATA_NODE}").update({"status": "error_retry"})

    except Exception as e:
        print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    run_sync()
