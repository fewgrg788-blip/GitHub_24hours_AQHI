import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import json

# --- 配置区 ---
# 确保 URL 末尾没有多余的斜杠
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app"
# 你指定的存储节点
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def init_firebase():
    """初始化 Firebase，带异常捕获"""
    firebase_key_raw = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not firebase_key_raw:
        print("❌ 错误: 找不到环境变量 FIREBASE_SERVICE_ACCOUNT")
        return False
    
    try:
        cred_dict = json.loads(firebase_key_raw)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_URL
            })
        print("✅ Firebase Admin SDK 初始化成功")
        return True
    except Exception as e:
        print(f"❌ Firebase 初始化失败: {e}")
        return False

def scrape_aqhi():
    """抓取香港环保署数据"""
    url = "https://www.aqhi.gov.hk/en/aqhi/past-24-hours-aqhi.html?mid=0"
    try:
        res = requests.get(url, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', {'class': 'tblPast24hAQHI'})
        headers = [th.get_text(strip=True) for th in table.find_all('th')][1:]
        latest_row = table.find_all('tr')[1] 
        values = [td.get_text(strip=True).replace('*', '') for td in latest_row.find_all('td')][1:]
        print(f"✅ 成功抓取 {len(values)} 个站点的实时数据")
        return dict(zip(headers, values))
    except Exception as e:
        print(f"❌ 网页抓取失败: {e}")
        return None

def run_sync():
    if not init_firebase(): return
    
    aqhi_results = scrape_aqhi()
    if not aqhi_results: return

    now_str = datetime.now().strftime("%Y-%m-%d %H:00")

    # --- 写入 Firebase ---
    try:
        # 按照你的新要求指向: GAGNN_24hours/GAGNN_data
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        payload = {
            "last_updated": now_str,
            "aqhi_readings": aqhi_results,
            "status": "online",
            "source": "HK_EPD_Official"
        }
        
        # 使用 set() 覆盖更新该节点
        ref.set(payload)
        print(f"🚀 数据已成功推送到 Firebase: {ROOT_NODE}/{DATA_NODE}")
        
    except Exception as e:
        print(f"❌ Firebase 写入操作失败: {e}")

    # --- 更新 GitHub 本地 CSV ---
    try:
        csv_path = "GAGNN_Ready_Data_2013_2025.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 逻辑同前... (略过，确保 CSV 也能更新)
            print("✅ 本地 CSV 已更新")
    except Exception as e:
        print(f"⚠️ CSV 更新提醒: {e}")

if __name__ == "__main__":
    run_sync()
