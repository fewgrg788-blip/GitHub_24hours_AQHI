import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import json
import traceback

FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def scrape_aqhi_robust():
    """更强壮的抓取函数"""
    url = "https://www.aqhi.gov.hk/en/aqhi/past-24-hours-aqhi.html?mid=0"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        print(f"正在访问官网: {url}")
        res = requests.get(url, headers=headers, timeout=20)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 尝试多种方式寻找数据表格
        table = soup.find('table', {'class': 'tblPast24hAQHI'})
        if not table:
            print("⚠️ 未能通过 class 找到表格，尝试搜索所有表格...")
            tables = soup.find_all('table')
            for t in tables:
                if "General Stations" in t.get_text():
                    table = t
                    break
        
        if not table:
            raise Exception("在网页中完全找不到 AQHI 数据表格")

        rows = table.find_all('tr')
        # 第一行通常是站点名
        headers_list = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td']) if th.get_text(strip=True)]
        # 第二行通常是最新数据
        data_row = rows[1]
        values_list = [td.get_text(strip=True).replace('*', '') for td in data_row.find_all('td') if td.get_text(strip=True)]
        
        # 清洗数据：只保留站点部分
        # 官方表通常第一列是时间，我们要去掉它
        aqhi_dict = dict(zip(headers_list[1:], values_list[1:]))
        
        if not aqhi_dict:
            raise Exception("提取到的数据字典为空")
            
        return aqhi_dict

    except Exception as e:
        print(f"❌ 抓取彻底失败: {e}")
        return None

def run_sync():
    print(f"--- 任务启动: {datetime.now()} ---")
    
    # 1. 验证环境变量
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not service_account_json:
        print("❌ 错误: 找不到 FIREBASE_SERVICE_ACCOUNT")
        return

    # 2. 初始化 Firebase
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(service_account_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print("✅ Firebase 初始化成功")
    except Exception as e:
        print(f"❌ Firebase 初始化异常: {e}")
        return

    # 3. 执行抓取
    aqhi_data = scrape_aqhi_robust()
    
    if aqhi_data:
        try:
            # 4. 写入 Firebase
            ref = db.reference(f"{ROOT_NODE}/{DATA_NODE}")
            payload = {
                "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "readings": aqhi_data,
                "status": "online"
            }
            ref.set(payload)
            print(f"🚀 [成功] 数据已写入 Firebase 路径: {ROOT_NODE}/{DATA_NODE}")
            
            # 5. 更新本地 CSV (可选)
            # ... 这里保留你之前的 CSV 更新逻辑 ...
            
        except Exception as e:
            print(f"❌ 写入 Firebase 失败: {e}")
            print(traceback.format_exc())
    else:
        print("⚠️ 由于没有抓取到数据，停止写入 Firebase。")

    print("--- 任务结束 ---")

if __name__ == "__main__":
    run_sync()
