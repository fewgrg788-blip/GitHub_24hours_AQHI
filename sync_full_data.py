import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# --- 配置区 ---
MASTER_CSV = "GAGNN_Ready_Data_2013_2025.csv"
HKT = timezone(timedelta(hours=8))

# AQHI 站点名称映射 (Data.HK -> 你的 CSV 列名)
AQHI_MAP = {
    "Central/Western": "AQHI_Central/Western", "Eastern": "AQHI_Eastern", 
    "Kwun Tong": "AQHI_Kwun Tong", "Sham Shui Po": "AQHI_Sham Shui Po",
    "Kwai Chung": "AQHI_Kwai Chung", "Tsuen Wan": "AQHI_Tsuen Wan",
    "Yuen Long": "AQHI_Yuen Long", "Tuen Mun": "AQHI_Tuen Mun",
    "Tung Chung": "AQHI_Tung Chung", "Tai Po": "AQHI_Tai Po",
    "Sha Tin": "AQHI_Sha Tin", "Tap Mun": "AQHI_Tap Mun",
    "Causeway Bay": "AQHI_Causeway Bay", "Central": "AQHI_Central",
    "Mong Kok": "AQHI_Mong Kok", "Tseung Kwan O": "AQHI_Tseung Kwan O",
    "Southern": "AQHI_Southern", "North": "AQHI_North"
}

# 湿度站点名称映射 (HKO 地名 -> 代码)
HUM_MAP = {
    "Hong Kong Observatory": "HKO", "King's Park": "KP", "Wong Chuk Hang": "YCT",
    "Ta Kwu Ling": "TKL", "Lau Fau Shan": "LFS", "Sha Tin": "SHA",
    "Tseung Kwan O": "JKB", "Sai Kung": "SKG", "Cheung Chau": "CCH",
    "Chek Lap Kok": "HKA", "Tsing Yi": "TYW", "Shek Kong": "SEK",
    "Tsuen Wan Ho Koon": "TMS", "Stanley": "SSH", "Peng Chau": "PEN",
    "Kowloon City": "KSC", "Tate's Cairn": "TC"
}

def fetch_daily_features():
    timestamp_str = datetime.now(HKT).strftime("%Y-%m-%d")
    row_data = {'Date': timestamp_str}

    # 1. 抓取 AQHI
    try:
        aqhi_res = requests.get("https://www.aqhi.gov.hk/epd/ddata/html/out/aqhi_ind_Eng.json", timeout=10).json()
        for item in aqhi_res:
            st_name = item.get('station')
            if st_name in AQHI_MAP:
                row_data[AQHI_MAP[st_name]] = item.get('aqhi')
    except Exception as e:
        print(f"AQHI Error: {e}")

    # 2. 抓取 Cyclone (气旋警告)
    row_data['Cyclone_Present'] = 0
    try:
        warns = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en", timeout=10).json()
        for w in warns.get('warningStatement', []):
            if "Tropical Cyclone" in w:
                row_data['Cyclone_Present'] = 1
                break
    except: pass

    # 3. 抓取 湿度 (HUM)
    try:
        hum_res = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en", timeout=10).json()
        for item in hum_res.get('humidity', {}).get('data', []):
            place = item.get('place')
            if place in HUM_MAP:
                code = HUM_MAP[place]
                row_data[f'HUM_{code}'] = item.get('value')
    except: pass

    # 4. 抓取 风速风向 (WSPD, PDIR)
    try:
        wind_res = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=stnwind&lang=en", timeout=10).json()
        for item in wind_res.get('stnList', []):
            code = item.get('stnCode')
            row_data[f'WSPD_{code}'] = item.get('windSpeed')
            row_data[f'PDIR_{code}'] = item.get('windDirection')
    except: pass

    return row_data

def append_to_master():
    print(f"--- 启动每日数据收集: {datetime.now(HKT)} ---")
    new_data = fetch_daily_features()
    
    # 读取主表的列结构
    if not os.path.isfile(MASTER_CSV):
        print(f"❌ 找不到主文件 {MASTER_CSV}")
        return
        
    master_cols = pd.read_csv(MASTER_CSV, nrows=0).columns.tolist()
    
    # 创建 DataFrame 并根据主表重新排序列，缺失的特征自动补 NaN
    df_new = pd.DataFrame([new_data]).reindex(columns=master_cols)
    
    # 追加到文件末尾
    df_new.to_csv(MASTER_CSV, mode='a', header=False, index=False)
    print(f"✅ 成功将 {new_data['Date']} 的数据追加到 {MASTER_CSV}")

if __name__ == "__main__":
    append_to_master()
