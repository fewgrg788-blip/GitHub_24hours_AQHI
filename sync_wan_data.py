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

# --- 1. 配置信息 (请仔细核对 URL) ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"
ROOT_NODE = "GAGNN_24hours"
DATA_NODE = "GAGNN_data"

def run_sync():
    print(f"--- 任务开始: {datetime.now()} ---")
    
    # A. 检查环境变量
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not service_account_json:
        print("❌ 错误: 找不到 GitHub Secret 'FIREBASE_SERVICE_ACCOUNT'。请检查 Secret 名称是否完全一致。")
        return

    # B. 初始化 Firebase
    try:
        print("正在尝试初始化 Firebase Admin SDK...")
        cred_dict = json.loads(service_account_json)
        cred = credentials.Certificate(cred_dict)
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_URL
            })
        print("✅ Firebase 初始化成功！")
    except Exception as e:
        print(f"❌ Firebase 初始化失败: {e}")
        return

    # C. 抓取数据
    print("正在抓取官网 AQHI 数据...")
    aqhi_data = None
    try:
        url = "https://www.aqhi.gov.hk/en/aqhi/past-24-hours-aqhi.html?mid=0"
        res = requests.get(url, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', {'class': 'tblPast24hAQHI'})
        
        # 简单提取最新一行
        headers = [th.get_text(strip=True) for th in table.find_all('th')][1:]
        latest_row = table.find_all('tr')[1] 
        values = [td.get_text(strip=True).replace('*', '') for td in latest_row.find_all('td')][1:]
        aqhi_data = dict(zip(headers, values))
        print(f"✅ 成功抓取数据，站点数量: {len(aqhi_data)}")
    except Exception as e:
        print(f"❌ 抓取失败: {e}")

    # D. 写入 Firebase (关键排错区)
    if aqhi_data:
        try:
            target_path = f"{ROOT_NODE}/{DATA_NODE}"
            print(f"正在尝试写入路径: {target_path} ...")
            
            ref = db.reference(target_path)
            
            # 构建上传数据
            payload = {
                "last_sync_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "readings": aqhi_data,
                "status": "active"
            }
            
            # 执行写入
            ref.set(payload)
            print("🚀 [SUCCESS] 数据已成功送达 Firebase！")
            
            # 测试读取一下确认写入成功
            check_data = ref.get()
            if check_data:
                print("验证成功: Firebase 已存在刚写入的数据。")
                
        except Exception as e:
            print(f"❌ [FIREBASE ERROR] 写入失败！详细原因:")
            print(traceback.format_exc()) # 打印完整错误堆栈

    # E. 更新 CSV (本地操作)
    print("正在检查本地 CSV 更新...")
    # ... (你的 CSV 更新逻辑) ...

    print(f"--- 任务结束 ---")

if __name__ == "__main__":
    run_sync()
