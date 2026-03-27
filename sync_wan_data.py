import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import json

# --- 1. 初始化 Firebase ---
# 从环境变量读取 Secret (由 GitHub Action 提供)
firebase_key_raw = os.getenv("FIREBASE_SERVICE_ACCOUNT")
if firebase_key_raw:
    key_dict = json.loads(firebase_key_raw)
    cred = credentials.Certificate(key_dict)
else:
    # 本地测试逻辑
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://project-12cc8-default-rtdb.firebaseio.com/'
    })

def scrape_hk_official_aqhi():
    """解析香港环保署官网 18 站点的实时 AQHI"""
    url = "https://www.aqhi.gov.hk/en/aqhi/past-24-hours-aqhi.html?mid=0"
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'tblPast24hAQHI'})
        
        # 获取所有站点名称（表头）
        headers = [th.get_text(strip=True) for th in table.find_all('th')][1:]
        
        # 获取最新一行数据（通常是第一行数据行）
        latest_row = table.find_all('tr')[1] 
        values = [td.get_text(strip=True).replace('*', '') for td in latest_row.find_all('td')][1:]
        
        # 转换为浮点数处理 (处理可能存在的空值)
        cleaned_values = []
        for v in values:
            try: cleaned_values.append(float(v))
            except: cleaned_values.append(np.nan)
            
        return dict(zip(headers, cleaned_values))
    except Exception as e:
        print(f"Scrape Error: {e}")
        return None

def run_sync():
    CSV_PATH = "GAGNN_Ready_Data_2013_2025.csv"
    data_map = scrape_hk_official_aqhi()
    if not data_map: return

    now_str = datetime.now().strftime("%Y-%m-%d %H:00")

    # A. 更新 Firebase (用于 BuildTech_System 实时渲染)
    ref = db.reference("BuildTech_System/WAN_GAGNN")
    ref.update({
        "Last_Sync": now_str,
        "Current_AQHI": data_map
    })

    # B. 追加到 CSV (用于模型学习)
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        new_entry = {"Date": now_str}
        # 匹配列名追加数据
        for col in df.columns:
            if col.startswith("AQHI_"):
                st_name = col.replace("AQHI_", "")
                # 模糊匹配：处理官网命名与 CSV 列名的微小差异
                matched_val = data_map.get(st_name, np.nan)
                new_entry[col] = matched_val
        
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        # 只保留最后 10000 行防止 CSV 过大，或保留全部
        df.to_csv(CSV_PATH, index=False)
        print(f"Successfully synced: {now_str}")

if __name__ == "__main__":
    run_sync()
