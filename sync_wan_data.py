import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import json

# --- 1. 初始化 Firebase (使用你的新 URL) ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app"
device_id = "56214328"

# 从 GitHub Secret 读取认证信息
firebase_key_raw = os.getenv("FIREBASE_SERVICE_ACCOUNT")
if firebase_key_raw:
    cred = credentials.Certificate(json.loads(firebase_key_raw))
else:
    # 本地测试请确保有此文件
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_URL
    })

def scrape_official_aqhi():
    """抓取官网当前 18 个站点的最新数据"""
    url = "https://www.aqhi.gov.hk/en/aqhi/past-24-hours-aqhi.html?mid=0"
    try:
        res = requests.get(url, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', {'class': 'tblPast24hAQHI'})
        
        # 获取表头 (站点名)
        headers = [th.get_text(strip=True) for th in table.find_all('th')][1:]
        # 获取最新一行数据
        latest_row = table.find_all('tr')[1] 
        values = [td.get_text(strip=True).replace('*', '') for td in latest_row.find_all('td')][1:]
        
        return dict(zip(headers, values))
    except Exception as e:
        print(f"Scrape Error: {e}")
        return None

def run():
    # A. 抓取数据
    aqhi_data = scrape_official_aqhi()
    if not aqhi_data: return
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:00")

    # B. 核心：在设备路径下更新 GAGNN_24hours 栏目
    # 路径：56214328/GAGNN_24hours
    ref = db.reference(f"/{device_id}/GAGNN_24hours")
    
    # 我们先存入当前的实测值作为基础，之后 Render 会更新预测数组
    ref.set({
        "last_sync": now_str,
        "current_official_data": aqhi_data,
        "status": "Waiting for GAGNN Inference",
        "GAGNN_forecast": [] # 留给 Render 填入未来 24 小时预测
    })

    # C. 追加到本地 CSV
    CSV_PATH = "GAGNN_Ready_Data_2013_2025.csv"
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        new_row = {"Date": now_str}
        # 匹配 AQHI_ 列
        for col in df.columns:
            if col.startswith("AQHI_"):
                st_name = col.replace("AQHI_", "")
                new_row[col] = aqhi_data.get(st_name, np.nan)
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

    print(f"✅ Firebase Column '{device_id}/GAGNN_24hours' has been updated.")

if __name__ == "__main__":
    run()
