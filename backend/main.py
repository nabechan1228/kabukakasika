from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
import os
import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
# 1.5. LSTMモデル定義
# ---------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        predictions = self.linear(out[:, -1, :])
        return predictions

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

        # RSI (14日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

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
                "rsi": clean_val(row['RSI']),
                "macd": clean_val(row['MACD']),
                "macdSignal": clean_val(row['Signal_Line']),
            })

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")
    finally:
        db.close()

@app.get("/api/predict/{code}")
def predict_stock(code: str):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    
    model_path = f"lstm_model_{code}.pth"
    scaler_path = f"scaler_{code}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"code": code, "prediction": None, "message": "学習モデルがありません"}

    db = SessionLocal()
    try:
        # 最新60件のデータを取得 (降順で取得して、後で昇順に戻す)
        query = f"SELECT close FROM stock_prices WHERE code = '{code}' ORDER BY date DESC LIMIT 60"
        df = pd.read_sql(query, con=engine)
        if len(df) < 60:
            return {"code": code, "prediction": None, "message": "予測に必要なデータ(60日分)が不足しています"}
        
        df = df.iloc[::-1].reset_index(drop=True)
        prices = df['close'].values.reshape(-1, 1)

        # スケーラー読み込み
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # モデル読み込み
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # データ前処理
        scaled_prices = scaler.transform(prices)
        X_tensor = torch.tensor(scaled_prices, dtype=torch.float32).unsqueeze(0)

        # 予測
        with torch.no_grad():
            pred_scaled = model(X_tensor)
        
        # スケールを元に戻す
        pred_actual = scaler.inverse_transform(pred_scaled.numpy())
        predicted_price = float(pred_actual[0][0])

        return {"code": code, "prediction": predicted_price, "message": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/api/train/{code}")
def train_stock_model(code: str):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    
    db = SessionLocal()
    try:
        # DBからデータを読み込む
        query = f"SELECT date, close FROM stock_prices WHERE code = '{code}' ORDER BY date ASC"
        df = pd.read_sql(query, con=engine)

        # DBにデータがない、または不足している場合は yfinance から取得して保存
        if df.empty or len(df) < 100:
            print(f"[{code}] DBにデータが不足しているため、yfinanceから取得します...")
            ticker_symbol = f"{code}.T"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                raise HTTPException(status_code=404, detail="株価データが見つかりません")

            # まとめてDBに保存
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
            db.bulk_save_objects(records)
            db.commit()
            
            # 再度読み込み
            df = pd.read_sql(query, con=engine)

        if len(df) < 100:
            raise HTTPException(status_code=400, detail="学習に必要なデータ（最低100日分）が不足しています")

        prices = df['close'].values.reshape(-1, 1)

        # 正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # スケーラー保存
        scaler_path = f"scaler_{code}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # シーケンスデータの作成 (過去60日 -> 翌日)
        SEQ_LENGTH = 60
        xs, ys = [], []
        for i in range(len(scaled_prices) - SEQ_LENGTH):
            xs.append(scaled_prices[i:(i + SEQ_LENGTH)])
            ys.append(scaled_prices[i + SEQ_LENGTH])
        X = np.array(xs)
        y = np.array(ys)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # LSTMモデル構築
        model = StockLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # オンデマンドでのレスポンス速度向上のため、高速学習エポック数（40）に設定
        EPOCHS = 40
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # モデル保存
        model_path = f"lstm_model_{code}.pth"
        torch.save(model.state_dict(), model_path)

        # 予測
        model.eval()
        last_60_scaled = scaled_prices[-SEQ_LENGTH:]
        X_pred_tensor = torch.tensor(last_60_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(X_pred_tensor)
        
        pred_actual = scaler.inverse_transform(pred_scaled.numpy())
        predicted_price = float(pred_actual[0][0])

        return {
            "code": code,
            "prediction": predicted_price,
            "message": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/info/{code}")
def get_stock_info(code: str):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    try:
        ticker = yf.Ticker(f"{code}.T")
        info = ticker.info
        return {
            "code": code,
            "name": info.get("longName") or info.get("shortName") or "不明",
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "previousClose": info.get("previousClose")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)