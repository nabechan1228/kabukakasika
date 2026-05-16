from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# ---------------------------------------------------------
# 1. データベースの初期設定 (SQLite x SQLAlchemy)
# ---------------------------------------------------------
DATABASE_URL = "sqlite:///./stock_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# テーブル定義（株価データ）
class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(String, index=True)  # 銘柄コード (例: 7203)
    date = Column(String, index=True)  # 日付 (YYYY-MM-DD)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

# 起動時にテーブルを作成
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------
# 2. FastAPI アプリケーション設定
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Stock API Server with SQLite is running!"}

# ---------------------------------------------------------
# 3. APIエンドポイント
# ---------------------------------------------------------
@app.get("/api/stock/{code}")
def get_stock_data(code: str):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードは4桁の数字で入力してください")

    db = SessionLocal()
    try:
        # DBからPandas DataFrameとして直接読み込む
        query = f"SELECT date, open, high, low, close, volume FROM stock_prices WHERE code = '{code}' ORDER BY date ASC"
        df = pd.read_sql(query, con=engine)

        # DBにデータが全くない場合は yfinance から初回取得して保存
        if df.empty:
            print(f"[{code}] DBにデータなし。yfinanceから取得します...")
            ticker_symbol = f"{code}.T"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1y") # 初回は過去1年分を取得

            if hist.empty:
                raise HTTPException(status_code=404, detail="データが見つかりませんでした")
            
            # DB保存用のリストを作成
            records = []
            for index, row in hist.iterrows():
                date_str = index.strftime('%Y-%m-%d')
                records.append(
                    StockPrice(
                        code=code,
                        date=date_str,
                        open=round(row['Open'], 1),
                        high=round(row['High'], 1),
                        low=round(row['Low'], 1),
                        close=round(row['Close'], 1),
                        volume=int(row['Volume'])
                    )
                )
            # まとめてDBに保存（バルクインサート）
            db.bulk_save_objects(records)
            db.commit()
            
            # 保存したデータを再度読み込み
            df = pd.read_sql(query, con=engine)

        # ---------------------------------------------------------
        # 指標の計算 (取得元がDBでも処理は同じ)
        # ---------------------------------------------------------
        # Pandasの機能を使うため、列名を統一
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_25'] = df['Close'].rolling(window=25).mean()
        std_25 = df['Close'].rolling(window=25).std()
        df['BB_Upper'] = df['SMA_25'] + (std_25 * 2)
        df['BB_Lower'] = df['SMA_25'] - (std_25 * 2)

        # 直近60日分に絞る
        df = df.tail(60)
        df = df.where((pd.notna(df)), None)

        # フロントエンド向けにJSON整形
        data = []
        for _, row in df.iterrows():
            def clean_val(val):
                return round(val, 1) if pd.notna(val) else None

            data.append({
                "date": row['date'],
                "open": clean_val(row['Open']),
                "high": clean_val(row['High']),
                "low": clean_val(row['Low']),
                "close": clean_val(row['Close']),
                "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0,
                "sma5": clean_val(row['SMA_5']),
                "sma25": clean_val(row['SMA_25']),
                "bbUpper": clean_val(row['BB_Upper']),
                "bbLower": clean_val(row['BB_Lower']),
            })

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)