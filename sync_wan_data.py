import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import json

# --- 配置 ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def scrape_real_aqhi():
    """抓取香港环保署数据，增加异常处理"""
    url = "https://www.aqhi.gov.hk/en/aqhi/past-24-hours-aqhi.html?mid=0"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        # 寻找数据表
        table = soup.find('table', {'class': 'tblPast24hAQHI'})
        if not table: return None
        
        rows = table.find_all('tr')
        # 提取表头（站点名）和第一行数据（最新）
        headers_list = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
        values_list = [td.get_text(strip=True).replace('*', '') for td in rows[1].find_all('td')]
        
        # 组合成字典，跳过第一列的时间
        data = dict(zip(headers_list[1:], values_list[1:]))
        return data
    except:
        return None

def run_sync():
    print(f"--- 启动任务: {datetime.now()} ---")
    
    # 1. 认证
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json: return
    
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        # 2. 抓取真实数据
        aqhi_results = scrape_real_aqhi()
        
        # 3. 写入 Firebase
        ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
        
        if aqhi_results:
            payload = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "aqhi_readings": aqhi_results,
                "status": "online"
            }
            ref.set(payload)
            print("🚀 [SUCCESS] 实时 AQHI 数据已同步到 Firebase！")
        else:
            # 如果抓取失败，至少更新心跳状态
            ref.update({"status": "scrape_failed", "last_attempt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            print("⚠️ 抓取失败，已更新 Firebase 状态。")

    except Exception as e:
        print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    run_sync()
