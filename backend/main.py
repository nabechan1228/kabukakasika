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
import threading
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

class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(String, index=True)
    date = Column(String, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# 1.5. マルチフィーチャーLSTMモデル (精度改善版 v2)
# ---------------------------------------------------------
FEATURE_COUNT = 5  # [close, volume_norm, sma5_dev, rsi, macd]
SEQ_LENGTH = 60

class StockLSTM(nn.Module):
    """v2: マルチフィーチャー対応 + Dropout + 2層LSTM"""
    def __init__(self, input_size=FEATURE_COUNT, hidden_size=128, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ---------------------------------------------------------
# 1.6. 特徴量エンジニアリング
# ---------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """close, volume カラムから5次元の特徴量を計算"""
    sma5 = df['close'].rolling(window=5).mean()
    df['sma5_dev'] = ((df['close'] - sma5) / sma5) * 100

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss_s
    df['rsi'] = 100 - (100 / (1 + rs))

    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26

    vol_max = df['volume'].max()
    df['volume_norm'] = df['volume'] / vol_max if vol_max > 0 else 0
    return df

def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = ['close', 'volume_norm', 'sma5_dev', 'rsi', 'macd']
    return df[cols].dropna().reset_index(drop=True).values

# ---------------------------------------------------------
# 1.8. スレッド安全な学習ステータス管理
# ---------------------------------------------------------
_training_lock = threading.Lock()
training_status = {}

def update_status(code: str, data: dict):
    with _training_lock:
        training_status[code] = data

def get_status_safe(code: str) -> dict:
    with _training_lock:
        return training_status.get(code, {
            "status": "idle", "progress": 0, "message": "未学習"
        }).copy()

# ---------------------------------------------------------
# 2. FastAPI アプリケーション設定
# ---------------------------------------------------------
app = FastAPI()

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
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
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードは4桁の数字で入力してください")

    try:
        query = text("SELECT date, open, high, low, close, volume FROM stock_prices WHERE code = :code ORDER BY date ASC")
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn, params={"code": code})

        if df.empty:
            logger.info(f"[{code}] DBにデータなし。yfinanceから取得します...")
            ticker = yf.Ticker(f"{code}.T")
            hist = ticker.history(period="3y")

            if hist.empty:
                raise HTTPException(status_code=404, detail="データが見つかりませんでした")

            records = []
            for index, row in hist.iterrows():
                date_str = index.strftime('%Y-%m-%d')
                records.append(StockPrice(
                    code=code, date=date_str,
                    open=round(row['Open'], 1), high=round(row['High'], 1),
                    low=round(row['Low'], 1), close=round(row['Close'], 1),
                    volume=int(row['Volume'])
                ))
            db.bulk_save_objects(records)
            db.commit()

            with engine.connect() as conn:
                df = pd.read_sql(query, con=conn, params={"code": code})

        # 指標の計算
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)

        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_25'] = df['Close'].rolling(window=25).mean()
        std_25 = df['Close'].rolling(window=25).std()
        df['BB_Upper'] = df['SMA_25'] + (std_25 * 2)
        df['BB_Lower'] = df['SMA_25'] - (std_25 * 2)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss_s
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df = df.tail(60)
        df = df.where(pd.notna(df), None)

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock data for {code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="株価データの取得中に内部エラーが発生しました。")


@app.get("/api/predict/{code}")
def predict_stock(code: str):
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")

    model_path = f"lstm_model_{code}.pth"
    scaler_path = f"scaler_{code}.json"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "学習モデルがありません"}

    try:
        with open(scaler_path, "r") as f:
            si = json.load(f)

        model_ver = si.get("model_version", 1)
        input_size = si.get("input_size", 1)

        query = text("SELECT date, close, volume FROM stock_prices WHERE code = :code ORDER BY date DESC LIMIT 120")
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn, params={"code": code})

        if len(df) < SEQ_LENGTH + 30:
            return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "予測に必要なデータが不足しています"}

        df = df.iloc[::-1].reset_index(drop=True)

        # スケーラー復元
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.data_min_ = np.array(si["data_min"])
        scaler.data_max_ = np.array(si["data_max"])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        safe_range = np.where(scaler.data_range_ == 0, 1, scaler.data_range_)
        scaler.scale_ = 1.0 / safe_range
        scaler.min_ = 0.0 - scaler.data_min_ * scaler.scale_

        close_min, close_max = scaler.data_min_[0], scaler.data_max_[0]
        close_range = close_max - close_min

        if model_ver >= 2:
            df = compute_features(df)
            fm = extract_feature_matrix(df)
            if len(fm) < SEQ_LENGTH:
                return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "データ不足"}

            scaled = scaler.transform(fm)
            model = StockLSTM(input_size=input_size)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            seq = scaled[-SEQ_LENGTH:].copy()
            preds = []
            for _ in range(5):
                xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    p = model(xt).item()
                preds.append(p)
                nr = seq[-1].copy()
                nr[0] = p
                seq = np.vstack([seq[1:], nr])

            actuals = [(p * close_range + close_min) for p in preds]
        else:
            # v1 レガシー互換
            prices = df['close'].values.reshape(-1, 1)
            scaled_p = scaler.transform(prices[-SEQ_LENGTH:])
            model = StockLSTM(input_size=1, hidden_size=50, num_layers=1, dropout=0)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            seq = scaled_p.copy()
            preds_s = []
            for _ in range(5):
                xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pv = model(xt).item()
                preds_s.append(pv)
                seq = np.append(seq[1:], [[pv]], axis=0)
            actuals = scaler.inverse_transform(
                np.array(preds_s).reshape(-1, 1)
            ).flatten().tolist()

        return {
            "code": code,
            "prediction": float(actuals[0]),
            "predictions": [float(x) for x in actuals],
            "mape": si.get("val_mape"),
            "message": "success"
        }

    except Exception as e:
        logger.error(f"Error predicting for {code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="予測処理中に内部エラーが発生しました。")


# ---------------------------------------------------------
# 4. 非同期学習タスク (精度改善版 v2)
# ---------------------------------------------------------
def run_training_task(code: str):
    db = SessionLocal()
    try:
        update_status(code, {"status": "training", "progress": 5, "message": "株価データをロード中..."})

        query = text("SELECT date, close, volume FROM stock_prices WHERE code = :code ORDER BY date ASC")
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn, params={"code": code})

        # データ不足の場合 yfinance から3年分取得
        if df.empty or len(df) < 200:
            update_status(code, {"status": "training", "progress": 10, "message": "yfinanceから3年分のデータを取得中..."})
            hist = yf.Ticker(f"{code}.T").history(period="3y")

            if hist.empty:
                update_status(code, {"status": "failed", "progress": 0, "message": "株価データが見つかりませんでした。"})
                return

            # 重複防止
            with engine.connect() as conn:
                existing = set(pd.read_sql(
                    text("SELECT date FROM stock_prices WHERE code = :code"),
                    conn, params={"code": code}
                )['date'].tolist()) if not df.empty else set()

            records = []
            for idx, row in hist.iterrows():
                ds = idx.strftime('%Y-%m-%d')
                if ds in existing:
                    continue
                records.append(StockPrice(
                    code=code, date=ds,
                    open=round(row['Open'], 1), high=round(row['High'], 1),
                    low=round(row['Low'], 1), close=round(row['Close'], 1),
                    volume=int(row['Volume'])
                ))
            if records:
                db.bulk_save_objects(records)
                db.commit()

            with engine.connect() as conn:
                df = pd.read_sql(query, con=conn, params={"code": code})

        if len(df) < 100:
            update_status(code, {"status": "failed", "progress": 0, "message": "学習に必要なデータが不足しています。"})
            return

        update_status(code, {"status": "training", "progress": 15, "message": "特徴量を計算中..."})

        # 特徴量エンジニアリング
        df = compute_features(df)
        fm = extract_feature_matrix(df)

        if len(fm) < SEQ_LENGTH + 20:
            update_status(code, {"status": "failed", "progress": 0, "message": "特徴量計算後のデータが不足しています。"})
            return

        # 正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(fm)

        # シーケンスデータ作成
        xs, ys = [], []
        for i in range(len(scaled) - SEQ_LENGTH):
            xs.append(scaled[i:(i + SEQ_LENGTH)])
            ys.append(scaled[i + SEQ_LENGTH, 0])  # 終値のみがターゲット
        X = np.array(xs)
        y = np.array(ys).reshape(-1, 1)

        # Train/Val 分割 (80:20)
        sp = int(len(X) * 0.8)
        Xt, Xv = torch.tensor(X[:sp], dtype=torch.float32), torch.tensor(X[sp:], dtype=torch.float32)
        yt, yv = torch.tensor(y[:sp], dtype=torch.float32), torch.tensor(y[sp:], dtype=torch.float32)

        update_status(code, {"status": "training", "progress": 20,
            "message": f"モデル構築中 (データ: {len(X)}サンプル, 特徴量: {FEATURE_COUNT}次元)"})

        # モデル・オプティマイザ構築
        model = StockLSTM(input_size=FEATURE_COUNT)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        EPOCHS = 150
        PATIENCE = 15
        best_val = float('inf')
        wait = 0
        best_state = None

        model.train()
        for ep in range(EPOCHS):
            optimizer.zero_grad()
            out = model(Xt)
            tl = criterion(out, yt)
            tl.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = criterion(model(Xv), yv).item()
            model.train()

            scheduler.step(vl)

            if vl < best_val:
                best_val = vl
                wait = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1

            if (ep + 1) % 5 == 0 or ep == 0:
                prog = 20 + int(((ep + 1) / EPOCHS) * 70)
                lr = optimizer.param_groups[0]['lr']
                update_status(code, {"status": "training", "progress": min(prog, 90),
                    "message": f"Epoch {ep+1}/{EPOCHS} | Train: {tl.item():.5f} | Val: {vl:.5f} | LR: {lr:.6f}"})

            if wait >= PATIENCE:
                logger.info(f"[{code}] Early stopping at epoch {ep+1}")
                update_status(code, {"status": "training", "progress": 90,
                    "message": f"Early Stopping (Epoch {ep+1}) | Best Val: {best_val:.5f}"})
                break

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), f"lstm_model_{code}.pth")

        # MAPE 計算
        model.eval()
        with torch.no_grad():
            vp = model(Xv).numpy().flatten()

        c_min, c_max = scaler.data_min_[0], scaler.data_max_[0]
        c_range = c_max - c_min
        va = y[sp:].flatten() * c_range + c_min
        vpp = vp * c_range + c_min

        mask = va != 0
        mape = float(np.mean(np.abs((va[mask] - vpp[mask]) / va[mask])) * 100) if mask.any() else None

        # スケーラー保存 (v2)
        sdata = {
            "feature_names": ["close", "volume_norm", "sma5_dev", "rsi", "macd"],
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "input_size": FEATURE_COUNT,
            "model_version": 2,
            "val_mape": round(mape, 2) if mape else None
        }
        with open(f"scaler_{code}.json", "w") as f:
            json.dump(sdata, f)

        update_status(code, {"status": "training", "progress": 95, "message": "未来5日間の予測を生成中..."})

        # 未来5日間予測
        seq = scaled[-SEQ_LENGTH:].copy()
        plist = []
        for _ in range(5):
            xp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pv = model(xp).item()
            plist.append(pv)
            nr = seq[-1].copy()
            nr[0] = pv
            seq = np.vstack([seq[1:], nr])

        pa = [(p * c_range + c_min) for p in plist]

        update_status(code, {
            "status": "success", "progress": 100,
            "prediction": float(pa[0]),
            "predictions": [float(x) for x in pa],
            "mape": round(mape, 2) if mape else None,
            "message": "success"
        })
        logger.info(f"[{code}] 学習完了 MAPE: {mape:.2f}%" if mape else f"[{code}] 学習完了")

    except Exception as e:
        logger.error(f"Training error for {code}: {e}", exc_info=True)
        update_status(code, {"status": "failed", "progress": 0,
            "message": f"学習中にエラーが発生しました: {str(e)}"})
    finally:
        db.close()


@app.post("/api/train/{code}")
def train_stock_model(code: str, background_tasks: BackgroundTasks):
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")

    current = get_status_safe(code)
    if current.get("status") == "training":
        return {"code": code, "message": "学習は既に実行中です。", "status": "training"}

    update_status(code, {"status": "training", "progress": 0, "message": "学習タスクを初期化中..."})
    background_tasks.add_task(run_training_task, code)

    return {"code": code, "message": "学習タスクを開始しました。", "status": "training"}


@app.get("/api/train/status/{code}")
def get_train_status(code: str):
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    return get_status_safe(code)


@app.get("/api/info/{code}")
def get_stock_info(code: str):
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=400, detail="銘柄コードが不正です")
    try:
        info = yf.Ticker(f"{code}.T").info
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
        logger.error(f"Error fetching info for {code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="企業情報の取得中にエラーが発生しました。")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)