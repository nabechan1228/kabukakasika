from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

import pandas as pd

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
        
        # 過去半年（6mo）のデータを取得（指標計算のため多めに取得）
        hist = ticker.history(period="6mo")

        if hist.empty:
             raise HTTPException(status_code=404, detail="指定された銘柄のデータが見つかりませんでした")

        # ---------------------------------------------------------
        # 指標の計算 (Pandasを利用)
        # ---------------------------------------------------------
        # 1. 単純移動平均線（SMA: Simple Moving Average）
        hist['SMA_5'] = hist['Close'].rolling(window=5).mean()
        hist['SMA_25'] = hist['Close'].rolling(window=25).mean()

        # 2. ボリンジャーバンド (25日, ±2σ)
        std_25 = hist['Close'].rolling(window=25).std()
        hist['BB_Upper'] = hist['SMA_25'] + (std_25 * 2)
        hist['BB_Lower'] = hist['SMA_25'] - (std_25 * 2)

        # ---------------------------------------------------------
        # フロントエンド向けにデータ整形
        # ---------------------------------------------------------
        # 直近約3ヶ月（60営業日）分に絞る
        hist = hist.tail(60)

        # NaNをNoneに変換してJSONエラーを防ぐ
        hist = hist.where((pd.notna(hist)), None)

        data = []
        for index, row in hist.iterrows():
            def clean_val(val):
                # NaN または None の場合は None を返す
                if pd.isna(val):
                    return None
                return round(float(val), 1)

            data.append({
                "date": index.strftime('%Y-%m-%d'),
                "open": clean_val(row['Open']),
                "high": clean_val(row['High']),
                "low": clean_val(row['Low']),
                "close": clean_val(row['Close']),
                "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                # 追加指標
                "sma5": clean_val(row['SMA_5']),
                "sma25": clean_val(row['SMA_25']),
                "bbUpper": clean_val(row['BB_Upper']),
                "bbLower": clean_val(row['BB_Lower']),
            })

        return data

    except Exception as e:
         raise HTTPException(status_code=500, detail=f"データ取得エラー: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)