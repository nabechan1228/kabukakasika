from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI()

# ---------------------------------------------------------
# CORSの設定（重要）
# フロントエンド（別のポート）からのアクセスを許可するために必要です
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発中は全て許可（本番環境ではフロントエンドのURLに絞ります）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Stock API Server is running!"}

@app.get("/api/stock/{code}")
def get_stock_data(code: str):
    """
    指定された4桁の銘柄コードの株価データを取得して返すAPI
    例: GET /api/stock/7203
    """
    # 4桁の数字かどうかを簡易チェック
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードは4桁の数字で入力してください")

    # 日本株の仕様に合わせて末尾に ".T" を追加
    ticker_symbol = f"{code}.T"
    
    try:
        # yfinanceでデータを取得
        ticker = yf.Ticker(ticker_symbol)
        
        # 過去3ヶ月分の日足データを取得（期間は変更可能: 1mo, 3mo, 1y など）
        hist = ticker.history(period="3mo")

        # データが存在しない場合の処理
        if hist.empty:
             raise HTTPException(status_code=404, detail="指定された銘柄のデータが見つかりませんでした")

        # フロントエンドで使いやすいJSON形式（配列）に整形
        data = []
        for index, row in hist.iterrows():
            data.append({
                "date": index.strftime('%Y-%m-%d'), # 日付を文字列に変換
                "open": round(row['Open'], 1),
                "high": round(row['High'], 1),
                "low": round(row['Low'], 1),
                "close": round(row['Close'], 1),
                "volume": int(row['Volume'])
            })

        return data

    except Exception as e:
         raise HTTPException(status_code=500, detail=f"データ取得エラー: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)