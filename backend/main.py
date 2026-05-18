from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import torch
import torch.nn as nn
import json
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 0. ログの設定
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("kabukakasika")

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

# DBセッション管理用の依存関係 (Depends)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
# 1.8. グローバル非同期学習ステータス管理
# ---------------------------------------------------------
# { code: { status: "idle" | "training" | "success" | "failed", progress: int, message: str, prediction: float, predictions: list } }
training_status = {}

# ---------------------------------------------------------
# 2. FastAPI アプリケーション設定
# ---------------------------------------------------------
app = FastAPI()

# 環境変数からCORSオリジンを取得（デフォルトはローカル開発用）
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
logger.info(f"CORS Allowed Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
def get_stock_data(code: str, db: Session = Depends(get_db)):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードは4桁の数字で入力してください")

    try:
        # DBからPandas DataFrameとして直接読み込む (パラメータバインディング)
        query = text("SELECT date, open, high, low, close, volume FROM stock_prices WHERE code = :code ORDER BY date ASC")
        df = pd.read_sql(query, con=engine.connect(), params={"code": code})

        # DBにデータが全くない場合は yfinance から初回取得して保存
        if df.empty:
            logger.info(f"[{code}] DBにデータなし。yfinanceから取得します...")
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
            
            # 保存したデータを再度読み込み (パラメータバインディング)
            df = pd.read_sql(query, con=engine.connect(), params={"code": code})

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

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching stock data for {code}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="株価データの取得中に内部エラーが発生しました。")

@app.get("/api/predict/{code}")
def predict_stock(code: str, db: Session = Depends(get_db)):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    
    model_path = f"lstm_model_{code}.pth"
    scaler_path_json = f"scaler_{code}.json"
    scaler_path_pkl = f"scaler_{code}.pkl"

    if not os.path.exists(model_path) or (not os.path.exists(scaler_path_json) and not os.path.exists(scaler_path_pkl)):
        return {"code": code, "prediction": None, "predictions": [], "message": "学習モデルがありません"}

    try:
        # 最新60件のデータを取得 (降順で取得して、後で昇順に戻す。パラメータバインディング化)
        query = text("SELECT close FROM stock_prices WHERE code = :code ORDER BY date DESC LIMIT 60")
        df = pd.read_sql(query, con=engine.connect(), params={"code": code})
        if len(df) < 60:
            return {"code": code, "prediction": None, "predictions": [], "message": "予測に必要なデータ(60日分)が不足しています"}
        
        df = df.iloc[::-1].reset_index(drop=True)
        prices = df['close'].values.reshape(-1, 1)

        # スケーラー読み込み (安全なJSON読み込み ＆ 自動移行)
        scaler = MinMaxScaler(feature_range=(0, 1))
        if os.path.exists(scaler_path_json):
            with open(scaler_path_json, "r") as f:
                sd = json.load(f)
            scaler.data_min_ = np.array(sd["data_min"])
            scaler.data_max_ = np.array(sd["data_max"])
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.scale_ = (1.0 - 0.0) / scaler.data_range_
            scaler.min_ = 0.0 - scaler.data_min_ * scaler.scale_
        elif os.path.exists(scaler_path_pkl):
            logger.info(f"[{code}] 古いスケーラー形式(.pkl)を検知。JSON形式へ自動移行します...")
            import pickle
            with open(scaler_path_pkl, "rb") as f:
                scaler = pickle.load(f)
            # 即座に JSON 形式で保存
            sd_data = {
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist()
            }
            with open(scaler_path_json, "w") as f:
                json.dump(sd_data, f)
            # 古い pkl ファイルを削除して安全にする
            try:
                os.remove(scaler_path_pkl)
                logger.info(f"[{code}] 旧スケーラー(.pkl)の自動削除に成功しました。")
            except Exception as e:
                logger.warning(f"[{code}] 旧スケーラー(.pkl)の自動削除に失敗しました: {e}")

        # モデル読み込み
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # データ前処理
        scaled_prices = scaler.transform(prices)
        
        # 自己回帰ループによる未来5日間の予測
        predictions_scaled = []
        current_seq = scaled_prices.copy()

        for _ in range(5):
            X_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model(X_tensor)
            pred_val = pred_scaled.item()
            predictions_scaled.append(pred_val)
            # シーケンスをスライドして最新の予測値を末尾に追加
            current_seq = np.append(current_seq[1:], [[pred_val]], axis=0)

        # スケールを元に戻す
        predictions_actual = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten().tolist()
        predicted_price = float(predictions_actual[0])

        return {
            "code": code,
            "prediction": predicted_price,
            "predictions": predictions_actual,
            "message": "success"
        }

    except Exception as e:
        logger.error(f"Error predicting stock for {code}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="予測処理中に内部エラーが発生しました。")

def run_training_task(code: str):
    db = SessionLocal()
    try:
        training_status[code] = {"status": "training", "progress": 5, "message": "株価データをロード中..."}
        
        # DBからデータを読み込む (パラメータバインディング化)
        query = text("SELECT date, close FROM stock_prices WHERE code = :code ORDER BY date ASC")
        df = pd.read_sql(query, con=engine.connect(), params={"code": code})

        # DBにデータがない、または不足している場合は yfinance から取得して保存
        if df.empty or len(df) < 100:
            training_status[code] = {"status": "training", "progress": 15, "message": "DBにデータが不足しているため、yfinanceから追加取得中..."}
            ticker_symbol = f"{code}.T"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                training_status[code] = {"status": "failed", "progress": 0, "message": "株価データが見つかりませんでした。"}
                return

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
            df = pd.read_sql(query, con=engine.connect(), params={"code": code})

        if len(df) < 100:
            training_status[code] = {"status": "failed", "progress": 0, "message": "学習に必要なデータ（最低100日分）が不足しています。"}
            return

        prices = df['close'].values.reshape(-1, 1)

        # 正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # スケーラー保存 (JSON形式)
        scaler_path = f"scaler_{code}.json"
        scaler_data = {
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist()
        }
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f)

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

        # エポック数を 100 回に引き上げ、より精度の高い予測が可能に！
        EPOCHS = 100
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            # 10エポックごとに進捗状況を更新 (進捗範囲 20% 〜 95%)
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                progress_percent = 20 + int(((epoch + 1) / EPOCHS) * 75)
                training_status[code] = {
                    "status": "training",
                    "progress": progress_percent,
                    "message": f"AIモデル学習中: Epoch {epoch+1}/{EPOCHS} (Loss: {loss.item():.5f})"
                }

        # モデル保存
        model_path = f"lstm_model_{code}.pth"
        torch.save(model.state_dict(), model_path)

        # 予測 (未来5日間の自己回帰ループ予測)
        model.eval()
        last_60_scaled = scaled_prices[-SEQ_LENGTH:]
        
        predictions_scaled = []
        current_seq = last_60_scaled.copy()

        for _ in range(5):
            X_pred_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model(X_pred_tensor)
            pred_val = pred_scaled.item()
            predictions_scaled.append(pred_val)
            # シーケンスをスライド
            current_seq = np.append(current_seq[1:], [[pred_val]], axis=0)

        # スケールを元に戻す
        predictions_actual = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten().tolist()
        predicted_price = float(predictions_actual[0])

        training_status[code] = {
            "status": "success",
            "progress": 100,
            "prediction": predicted_price,
            "predictions": predictions_actual,
            "message": "success"
        }
        logger.info(f"[{code}] 非同期AIモデル学習が正常に完了しました。")

    except Exception as e:
        logger.error(f"Error in async training task for {code}: {str(e)}", exc_info=True)
        training_status[code] = {
            "status": "failed",
            "progress": 0,
            "message": f"学習中にエラーが発生しました: {str(e)}"
        }
    finally:
        db.close()

@app.post("/api/train/{code}")
def train_stock_model(code: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    
    # 既に学習中の場合は何もしない
    current = training_status.get(code, {})
    if current.get("status") == "training":
        return {"code": code, "message": "学習は既に実行中です。", "status": "training"}

    training_status[code] = {
        "status": "training",
        "progress": 0,
        "message": "学習タスクを初期化中..."
    }

    # 非同期で学習をバックグラウンド実行
    background_tasks.add_task(run_training_task, code)

    return {
        "code": code,
        "message": "学習タスクを開始しました。",
        "status": "training"
    }

@app.get("/api/train/status/{code}")
def get_train_status(code: str):
    if not code.isalnum() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    
    status = training_status.get(code, {"status": "idle", "progress": 0, "message": "未学習"})
    return status

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
        logger.error(f"Error fetching stock info for {code}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="企業情報の取得中に内部エラーが発生しました。")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)