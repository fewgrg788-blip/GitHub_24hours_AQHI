import os
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==========================================
# 1. 伺服器健康檢查 (Render 會自動訪問此路徑確保存活)
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "project": "BuildTech GAGNN",
        "message": "GAGNN API is running smoothly!"
    }), 200

# ==========================================
# 2. 接收同步數據 (解決 GitHub Action 報錯 404 的關鍵)
# ==========================================
@app.route('/sync', methods=['POST'])
def sync_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        print(f"📥 收到 GitHub Action 同步數據，時間: {data.get('Date')}")
        
        # [選項] 你可以在這裡加入把接收到的 JSON 轉寫入本地 CSV 的邏輯
        # 但如果你已經設定了 Deploy Hook，Render 會自動抓取 GitHub 最新檔案，
        # 所以這裡只要成功回傳 200 狀態碼，讓 GitHub Action 知道傳輸成功即可。
        
        return jsonify({
            "status": "success",
            "received_date": data.get("Date"),
            "message": "Data synchronized with Render node successfully!"
        }), 200

    except Exception as e:
        print(f"❌ 同步失敗: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 3. 輸出預測結果 (讓你的前端網頁或 App 讀取)
# ==========================================
@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        file_path = "gagnn_prediction_today.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 將 CSV 轉換為 JSON 格式回傳
            return jsonify({
                "status": "success", 
                "data": df.to_dict(orient="records")
            }), 200
        else:
            return jsonify({"error": "Prediction file not found. Model might still be running/training."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 伺服器啟動設定
# ==========================================
if __name__ == "__main__":
    # ⚠️ 關鍵：Render 雲端環境必須監聽 0.0.0.0，並讀取系統分配的 PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
