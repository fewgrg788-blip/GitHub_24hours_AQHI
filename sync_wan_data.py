import os
import json
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# --- 严格核对配置 ---
FIREBASE_URL = "https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app/"

def test_firebase_connection():
    print(f"--- 启动时间: {datetime.now()} ---")
    
    # 1. 检查环境变量
    creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not creds_json:
        print("❌ 错误: 环境变量 FIREBASE_SERVICE_ACCOUNT 为空")
        return

    try:
        # 2. 解析 JSON
        print("正在解析 Service Account JSON...")
        cert_info = json.loads(creds_json)
        
        # 3. 初始化
        print(f"正在连接数据库: {FIREBASE_URL}")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cert_info)
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        
        # 4. 强行写入测试数据 (不依赖爬虫)
        print("正在尝试写入测试心跳...")
        ref = db.reference("GAGNN_24hours/connection_test")
        test_payload = {
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "GitHub Action Connection OK",
            "message": "If you see this, Firebase connection is working!"
        }
        ref.set(test_payload)
        print("🚀 [SUCCESS] 测试数据已成功写入 Firebase！")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: 请检查 GitHub Secret 是否包含完整的 { 和 }。错误: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    test_firebase_connection()
