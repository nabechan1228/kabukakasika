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
# 1.5. マルチフィーチャーLSTMモデル (精度改善版 v4)
# ---------------------------------------------------------
FEATURE_COUNT = 7  # [log_return, volume_norm, sma5_dev, rsi, macd, n225_return, usdjpy_return]
SEQ_LENGTH = 60
PREDICT_DAYS = 5   # 未来5日間を一括予測

class StockLSTM(nn.Module):
    """v3: Multi-step出力(5日一括予測) + log_return特徴量 + 過学習対策
    output_size パラメータで v1/v2 モデルの後方互換読み込みにも対応"""
    def __init__(self, input_size=FEATURE_COUNT, hidden_size=64, num_layers=2, dropout=0.3, output_size=PREDICT_DAYS):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)  # 5日分を一括出力

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ---------------------------------------------------------
# 1.6. 特徴量エンジニアリング
# ---------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """close, volume カラムから特徴量を計算 (v3: log_return 追加)"""
    # 対数収益率 — 非定常性・データリーク問題を解決
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

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
    """v2互換: close 基準の特徴量行列"""
    cols = ['close', 'volume_norm', 'sma5_dev', 'rsi', 'macd']
    return df[cols].dropna().reset_index(drop=True).values

def extract_feature_matrix_v3(df: pd.DataFrame) -> np.ndarray:
    """v3: log_return 基準の特徴量行列（非定常性問題を解消）"""
    cols = ['log_return', 'volume_norm', 'sma5_dev', 'rsi', 'macd']
    return df[cols].dropna().reset_index(drop=True).values

def compute_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """v4: マクロ指標 (日経平均, ドル円) の変化率を追加"""
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

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

    # マクロ指標の取得と変化率計算
    import datetime
    start_date = df['date'].min()
    end_date_dt = datetime.datetime.strptime(df['date'].max(), '%Y-%m-%d') + datetime.timedelta(days=2)
    end_date = end_date_dt.strftime('%Y-%m-%d')
    
    # yfinance は警告を出さないように quiet=True でダウンロード（可能であれば）
    macro_n225 = yf.download("^N225", start=start_date, end=end_date, progress=False)
    macro_fx = yf.download("USDJPY=X", start=start_date, end=end_date, progress=False)
    
    n225_close = macro_n225['Close'].squeeze() if 'Close' in macro_n225 else macro_n225.iloc[:, 0]
    fx_close = macro_fx['Close'].squeeze() if 'Close' in macro_fx else macro_fx.iloc[:, 0]
    
    n225_return = n225_close.pct_change()
    fx_return = fx_close.pct_change()
    
    macro_df = pd.DataFrame({
        'n225_return': n225_return,
        'usdjpy_return': fx_return
    }).reset_index()
    
    date_col = 'Date' if 'Date' in macro_df.columns else macro_df.columns[0]
    macro_df['date'] = macro_df[date_col].dt.strftime('%Y-%m-%d')
    
    df = pd.merge(df, macro_df[['date', 'n225_return', 'usdjpy_return']], on='date', how='left')
    df['n225_return'] = df['n225_return'].fillna(0)
    df['usdjpy_return'] = df['usdjpy_return'].fillna(0)
    
    return df

def extract_feature_matrix_v4(df: pd.DataFrame) -> np.ndarray:
    """v4: 7次元特徴量行列"""
    cols = ['log_return', 'volume_norm', 'sma5_dev', 'rsi', 'macd', 'n225_return', 'usdjpy_return']
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

        if model_ver >= 4:
            # v4: 7次元特徴量 + マクロ指標
            df = compute_features_v4(df)
            fm = extract_feature_matrix_v4(df)
            if len(fm) < SEQ_LENGTH:
                return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "データ不足"}

            df_clean = df[['close', 'log_return', 'volume_norm', 'sma5_dev', 'rsi', 'macd', 'n225_return', 'usdjpy_return']].dropna().reset_index(drop=True)
            last_close = float(df_clean['close'].iloc[-1])

            scaled = scaler.transform(fm)
            model = StockLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3, output_size=PREDICT_DAYS)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            seq = scaled[-SEQ_LENGTH:].copy()
            xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                log_preds_scaled = model(xt).numpy().flatten()  # shape: (PREDICT_DAYS,)

            lr_range = close_range
            log_preds = log_preds_scaled * lr_range + close_min
            pa = []
            price = last_close
            for lr in log_preds:
                price = price * np.exp(float(lr))
                pa.append(price)
            actuals = pa

        elif model_ver == 3:
            # v3: log_return特徴量 + 1回の推論で5日分一括予測（特徴量崩壊なし）
            df = compute_features(df)
            fm = extract_feature_matrix_v3(df)
            if len(fm) < SEQ_LENGTH:
                return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "データ不足"}

            # close価格を feature matrix と同じ行インデックスで取得
            df_clean = df[['close', 'log_return', 'volume_norm', 'sma5_dev', 'rsi', 'macd']].dropna().reset_index(drop=True)
            last_close = float(df_clean['close'].iloc[-1])

            scaled = scaler.transform(fm)
            model = StockLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3, output_size=PREDICT_DAYS)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            seq = scaled[-SEQ_LENGTH:].copy()
            xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                log_preds_scaled = model(xt).numpy().flatten()  # shape: (PREDICT_DAYS,)

            # log_return を元スケールに戻して累積価格へ変換
            lr_range = close_range
            log_preds = log_preds_scaled * lr_range + close_min
            pa = []
            price = last_close
            for lr in log_preds:
                price = price * np.exp(float(lr))
                pa.append(price)
            actuals = pa

        elif model_ver >= 2:
            # v2 レガシー互換: close特徴量 + 自己回帰ループ
            df = compute_features(df)
            fm = extract_feature_matrix(df)
            if len(fm) < SEQ_LENGTH:
                return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "データ不足"}

            scaled = scaler.transform(fm)
            model = StockLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)
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
            model = StockLSTM(input_size=1, hidden_size=50, num_layers=1, dropout=0, output_size=1)
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

        update_status(code, {"status": "training", "progress": 15, "message": "特徴量とマクロ指標を計算中..."})

        # 特徴量エンジニアリング (v4: マクロ指標 使用)
        df = compute_features_v4(df)
        fm = extract_feature_matrix_v4(df)

        if len(fm) < SEQ_LENGTH + PREDICT_DAYS:
            update_status(code, {"status": "failed", "progress": 0, "message": "特徴量計算後のデータが不足しています。"})
            return

        # 正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(fm)

        # シーケンスデータ作成 (v3: 未来PREDICT_DAYS日分を一括ターゲット)
        xs, ys = [], []
        for i in range(len(scaled) - SEQ_LENGTH - PREDICT_DAYS + 1):
            xs.append(scaled[i:(i + SEQ_LENGTH)])
            # 未来PREDICT_DAYS日分の log_return (インデックス0) をターゲットに
            ys.append(scaled[i + SEQ_LENGTH : i + SEQ_LENGTH + PREDICT_DAYS, 0])
        X = np.array(xs)
        y = np.array(ys)  # shape: (N, PREDICT_DAYS)

        # Train/Val 分割 (80:20)
        sp = int(len(X) * 0.8)
        Xt, Xv = torch.tensor(X[:sp], dtype=torch.float32), torch.tensor(X[sp:], dtype=torch.float32)
        yt, yv = torch.tensor(y[:sp], dtype=torch.float32), torch.tensor(y[sp:], dtype=torch.float32)

        update_status(code, {"status": "training", "progress": 20,
            "message": f"モデル構築中 v4 (データ: {len(X)}サンプル, 入力: {FEATURE_COUNT}次元, 出力: {PREDICT_DAYS}日)"})

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

        # MAPE 計算 (v3: log_return ベース — 変化率の平均絶対誤差)
        model.eval()
        with torch.no_grad():
            vp = model(Xv).numpy()  # shape: (val_size, PREDICT_DAYS)

        c_min, c_max = scaler.data_min_[0], scaler.data_max_[0]
        c_range = c_max - c_min

        # log_return の MAE を % 換算で報告
        # (log_return ≈ 前日比変化率なので、MAE×100 = "平均絶対リターン誤差(%/日)" として解釈可能)
        # MAPEは log_return≈0 の日に爆発するため使用しない
        va_lr = y[sp:].flatten() * c_range + c_min
        vp_lr = vp.flatten() * c_range + c_min
        mae_pct = float(np.mean(np.abs(va_lr - vp_lr)) * 100)  # 例: 1.5 → "±1.5%/日の誤差"
        mape = round(mae_pct, 4) if not np.isnan(mae_pct) else None

        # スケーラー保存 (v4)
        last_close = float(df['close'].dropna().iloc[-1])
        sdata = {
            "feature_names": ["log_return", "volume_norm", "sma5_dev", "rsi", "macd", "n225_return", "usdjpy_return"],
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "input_size": FEATURE_COUNT,
            "model_version": 4,
            "last_close": last_close,
            "val_mape": round(mape, 2) if mape else None
        }
        with open(f"scaler_{code}.json", "w") as f:
            json.dump(sdata, f)

        update_status(code, {"status": "training", "progress": 95, "message": "未来5日間の予測を生成中..."})

        # 未来5日間予測 (v4: 1回の推論で一括取得、特徴量崩壊なし)
        seq = scaled[-SEQ_LENGTH:].copy()
        xp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            log_preds_scaled = model(xp).numpy().flatten()  # shape: (PREDICT_DAYS,)

        # log_return を元スケールに戻して累積価格へ変換
        log_preds = log_preds_scaled * c_range + c_min
        pa = []
        price = last_close
        for lr in log_preds:
            price = price * np.exp(float(lr))
            pa.append(price)

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