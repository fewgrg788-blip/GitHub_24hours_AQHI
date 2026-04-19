import datetime
import os

def save_to_annual_file(new_data_row, column_headers):
    # 根據當前年份決定存到哪個檔案
    current_year = datetime.datetime.now().year
    target_file = f"aqhi_history_{current_year}.csv"
    
    file_exists = os.path.isfile(target_file)
    
    # 以附加模式 (a) 寫入
    with open(target_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(column_headers) + "\n")
        f.write(",".join(map(str, new_data_row)) + "\n")
    print(f"✅ Data synced to {target_file}")
