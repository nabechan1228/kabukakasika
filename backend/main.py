"""
kabukakasika バックエンド - FastAPI メインアプリケーション

株価データの取得・保存、AI予測（Attention LSTM）、モデル学習のAPIを提供する。
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
import os
import datetime as dt
import torch
import json
import logging
import numpy as np
import threading
from sklearn.preprocessing import MinMaxScaler

# --- 共通モジュール ---
from db import engine, SessionLocal, StockPrice, get_db
from validators import validate_stock_code, get_model_path, get_scaler_path
from models import (
    StockAttentionLSTM, FEATURE_COUNT, SEQ_LENGTH, PREDICT_DAYS,
    restore_scaler, load_model
)
from features import compute_features, extract_feature_matrix

# ---------------------------------------------------------
# 0. ログの設定
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("kabukakasika")

# ---------------------------------------------------------
# 1. 最新日データの多層自己補完ヘルパー関数
# ---------------------------------------------------------
def safe_complement_historical_data(code: str, ticker: yf.Ticker, hist: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance のヒストリカルデータ hist の最終行（最新営業日）に NaN がある場合、
    1) period="1d" による最新クォートの取得
    2) ticker.info からのリアルタイム情報取得
    を順次試みて補完し、日付のズレがないように安全にマージ/上書きして返します。
    """
    if hist.empty:
        return hist
        
    last_idx = hist.index[-1]
    last_row = hist.iloc[-1]
    
    # 価格情報のいずれかが NaN の場合に補完処理に入る
    if pd.isna(last_row['Open']) or pd.isna(last_row['Close']) or pd.isna(last_row['High']) or pd.isna(last_row['Low']):
        try:
            logger.info(f"[{code}] 最新日 {last_idx.strftime('%Y-%m-%d')} のデータに欠損値があるため、補完処理を開始します。")
            
            today_row = None
            today_idx = None
            
            # 1. period="1d" での再取得を試みる
            try:
                today_hist = ticker.history(period="1d")
                if not today_hist.empty:
                    t_row = today_hist.iloc[-1]
                    if not (pd.isna(t_row['Open']) or pd.isna(t_row['Close']) or pd.isna(t_row['High']) or pd.isna(t_row['Low'])):
                        today_row = t_row
                        today_idx = today_hist.index[-1]
                        logger.info(f"[{code}] period='1d' から最新データを取得しました。")
            except Exception as ex:
                logger.warning(f"[{code}] period='1d' による補完試行中にエラー: {ex}")
                
            # 2. 取得できなかった場合、ticker.info からの取得を試みる
            if today_row is None:
                try:
                    info = ticker.info
                    o = info.get('open')
                    h = info.get('dayHigh')
                    l = info.get('dayLow')
                    c = info.get('currentPrice') or info.get('regularMarketPrice')
                    v = info.get('volume') or info.get('regularMarketVolume') or 0
                    
                    if o is not None and h is not None and l is not None and c is not None:
                        today_row = pd.Series({
                            'Open': float(o),
                            'High': float(h),
                            'Low': float(l),
                            'Close': float(c),
                            'Volume': int(v) if v is not None and not pd.isna(v) else 0
                        })
                        if isinstance(last_idx, pd.Timestamp) and last_idx.tz is not None:
                            today_idx = pd.Timestamp.now(tz=last_idx.tz).normalize()
                        else:
                            today_idx = pd.Timestamp.now().normalize()
                        logger.info(f"[{code}] ticker.info から最新データを取得しました。")
                except Exception as ex:
                    logger.warning(f"[{code}] ticker.info による補完試行中にエラー: {ex}")
            
            # 3. 取得できた最新データを用いて hist を上書き/追加補完する
            if today_row is not None and today_idx is not None:
                last_date_str = last_idx.strftime('%Y-%m-%d')
                today_date_str = today_idx.strftime('%Y-%m-%d')
                
                if last_date_str == today_date_str:
                    hist.loc[last_idx, 'Open'] = today_row['Open']
                    hist.loc[last_idx, 'High'] = today_row['High']
                    hist.loc[last_idx, 'Low'] = today_row['Low']
                    hist.loc[last_idx, 'Close'] = today_row['Close']
                    hist.loc[last_idx, 'Volume'] = today_row['Volume']
                    logger.info(f"[{code}] 最終行 {last_date_str} のデータを正常に補完（上書き）しました。")
                else:
                    tz_today_idx = today_idx
                    if isinstance(last_idx, pd.Timestamp) and last_idx.tz is not None:
                        tz_today_idx = today_idx.tz_convert(last_idx.tz) if today_idx.tz is not None else today_idx.tz_localize(last_idx.tz)
                    
                    if tz_today_idx not in hist.index:
                        new_row_df = pd.DataFrame([today_row], index=[tz_today_idx])
                        hist = pd.concat([hist, new_row_df])
                        logger.info(f"[{code}] 新たな日付 {today_date_str} の行を追加し、補完しました。")
                    else:
                        hist.loc[tz_today_idx, 'Open'] = today_row['Open']
                        hist.loc[tz_today_idx, 'High'] = today_row['High']
                        hist.loc[tz_today_idx, 'Low'] = today_row['Low']
                        hist.loc[tz_today_idx, 'Close'] = today_row['Close']
                        hist.loc[tz_today_idx, 'Volume'] = today_row['Volume']
                        logger.info(f"[{code}] 既存の行 {today_date_str} のデータを正常に補完（上書き）しました。")
        except Exception as e:
            logger.error(f"[{code}] 補完処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            
    return hist


# ---------------------------------------------------------
# 2. yfinance から取得した株価データをDBに保存するヘルパー
# ---------------------------------------------------------
def _is_row_complete(row) -> bool:
    """株価行データの価格情報が完全かどうかチェックする。"""
    return not (pd.isna(row['Open']) or pd.isna(row['Close']) or pd.isna(row['High']) or pd.isna(row['Low']))


def _hist_to_records(code: str, hist: pd.DataFrame, skip_before: str | None = None) -> list[StockPrice]:
    """
    yfinanceのヒストリカルDataFrameからStockPriceレコードのリストを生成する。

    Args:
        code: 銘柄コード
        hist: yfinanceのhistory結果
        skip_before: この日付以前のデータをスキップする（差分更新用）

    Returns:
        StockPriceオブジェクトのリスト
    """
    records = []
    for index, row in hist.iterrows():
        date_str = index.strftime('%Y-%m-%d')
        
        # 指定日以前のデータをスキップ
        if skip_before and date_str <= skip_before:
            continue
        
        # NaN が含まれている行は登録をスキップする
        if not _is_row_complete(row):
            logger.warning(f"[{code}] {date_str} の価格データが不完全なため、登録をスキップします。")
            continue
            
        records.append(StockPrice(
            code=code, date=date_str,
            open=round(row['Open'], 1), high=round(row['High'], 1),
            low=round(row['Low'], 1), close=round(row['Close'], 1),
            volume=int(row['Volume']) if pd.notna(row['Volume']) else 0
        ))
    return records


def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """APIレスポンス用のテクニカル指標（SMA, BB, RSI, MACD）を計算する。"""
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_25'] = df['close'].rolling(window=25).mean()
    std_25 = df['close'].rolling(window=25).std()
    df['BB_Upper'] = df['SMA_25'] + (std_25 * 2)
    df['BB_Lower'] = df['SMA_25'] - (std_25 * 2)

    # RSI 計算
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss_s
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD 計算
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df


# ---------------------------------------------------------
# 3. スレッド安全な学習ステータス管理
# ---------------------------------------------------------
_training_lock = threading.Lock()
training_status = {}


def update_status(code: str, data: dict):
    with _training_lock:
        training_status[code] = data


def get_status_safe(code: str) -> dict:
    with _training_lock:
        return training_status.get(code, {"status": "idle", "progress": 0, "message": "未学習"}).copy()


def _is_training(code: str) -> bool:
    """指定銘柄が現在学習中かどうかを確認する。"""
    with _training_lock:
        status = training_status.get(code, {})
        return status.get("status") == "training"


# ---------------------------------------------------------
# 4. FastAPI アプリケーション設定
# ---------------------------------------------------------
app = FastAPI(
    title="kabukakasika API",
    description="日本株の株価データ取得・AI予測API",
    version="2.0.0"
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 5. 株価データ取得 API
# ---------------------------------------------------------
STOCK_DATA_QUERY = text(
    "SELECT date, open, high, low, close, volume FROM stock_prices "
    "WHERE code = :code ORDER BY date ASC"
)


@app.get("/api/stock/{code}")
def get_stock_data(code: str, db: Session = Depends(get_db)):
    code = validate_stock_code(code)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(STOCK_DATA_QUERY, con=conn, params={"code": code})
        
        if df.empty:
            # 初回取得: yfinanceから3年分のデータを取得してDBに保存
            ticker = yf.Ticker(f"{code}.T")
            hist = ticker.history(period="3y")
            
            if not hist.empty:
                hist = safe_complement_historical_data(code, ticker, hist)
            
            records = _hist_to_records(code, hist)
            if records:
                db.bulk_save_objects(records)
                db.commit()
            
            with engine.connect() as conn:
                df = pd.read_sql(STOCK_DATA_QUERY, con=conn, params={"code": code})
        else:
            # 差分更新: 既存データがある場合、最新データのみ取得
            # 不完全なデータ（価格が空）を先にクリーンアップ
            db.execute(
                text("DELETE FROM stock_prices WHERE code = :code AND (open IS NULL OR close IS NULL)"),
                {"code": code}
            )
            db.commit()
            
            # 削除後に再度データを読み直して最新日を取得
            with engine.connect() as conn:
                df = pd.read_sql(STOCK_DATA_QUERY, con=conn, params={"code": code})
            
            last_date_str = df['date'].max()
            today = dt.datetime.now().strftime('%Y-%m-%d')
            
            if last_date_str <= today:
                ticker = yf.Ticker(f"{code}.T")
                hist = ticker.history(start=last_date_str)
                if not hist.empty:
                    hist = safe_complement_historical_data(code, ticker, hist)
                    records = _hist_to_records(code, hist, skip_before=last_date_str)
                    if records:
                        db.bulk_save_objects(records)
                        db.commit()
                        with engine.connect() as conn:
                            df = pd.read_sql(STOCK_DATA_QUERY, con=conn, params={"code": code})

        # テクニカル指標を計算
        df = _compute_technical_indicators(df)

        df = df.tail(60).where(pd.notna(df), None)
        
        data = []
        for _, row in df.iterrows():
            cv = lambda v: round(v, 1) if pd.notna(v) else None
            data.append({
                "date": row['date'],
                "open": cv(row['open']),
                "high": cv(row['high']),
                "low": cv(row['low']),
                "close": cv(row['close']),
                "volume": int(row['volume']) if pd.notna(row['volume']) else 0,
                "sma5": cv(row['SMA_5']),
                "sma25": cv(row['SMA_25']),
                "bbUpper": cv(row['BB_Upper']),
                "bbLower": cv(row['BB_Lower']),
                "rsi": cv(row['RSI']),
                "macd": cv(row['MACD']),
                "macdSignal": cv(row['Signal_Line'])
            })
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock {code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="株価データの取得中にエラーが発生しました。")


# ---------------------------------------------------------
# 6. AI 推論エンドポイント (変化率から実価格への復元)
# ---------------------------------------------------------
@app.get("/api/predict/{code}")
def predict_stock(code: str):
    code = validate_stock_code(code)
    
    model_path = get_model_path(code)
    scaler_path = get_scaler_path(code)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"code": code, "prediction": None, "predictions": [], "mape": None, "message": "学習モデルがありません"}

    try:
        # スケーラー復元（共通ヘルパー使用）
        try:
            scaler, si = restore_scaler(scaler_path)
        except ValueError as ve:
            return {
                "code": code, "prediction": None, "predictions": [],
                "mape": None, "message": str(ve)
            }

        # 最新データを多めに取得して前処理
        query = text("SELECT date, close, volume FROM stock_prices WHERE code = :code ORDER BY date DESC LIMIT 120")
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn, params={"code": code})
        
        df = df.iloc[::-1].reset_index(drop=True)
        # 復元用に最後の実際の「終値」を保存しておく
        last_actual_price = df['close'].iloc[-1]
        
        df = compute_features(df)
        fm = extract_feature_matrix(df)

        # ターゲットはインデックス0（close_return）
        ret_min, ret_max = scaler.data_min_[0], scaler.data_max_[0]
        ret_range = ret_max - ret_min

        # 推論
        scaled = scaler.transform(fm)
        seq = scaled[-SEQ_LENGTH:].copy()
        xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        model = load_model(model_path, input_size=si.get("input_size", FEATURE_COUNT))

        with torch.no_grad():
            preds_scaled_returns = model(xt).numpy().flatten()

        # スケールされた変化率を、実際の中身（例: 0.015 = 1.5%）に戻す
        actual_returns = [(p * ret_range + ret_min) for p in preds_scaled_returns]

        # 変化率の予測値から、未来の実価格を計算する
        predicted_prices = []
        current_price = last_actual_price
        for r in actual_returns:
            current_price = current_price * (1 + r)  # 前日価格 × (1 + 変化率)
            predicted_prices.append(round(current_price, 1))

        return {
            "code": code,
            "prediction": predicted_prices[0],
            "predictions": predicted_prices,
            "mape": si.get("val_mape"),
            "message": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting {code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="予測処理中にエラーが発生しました。")


# ---------------------------------------------------------
# 7. 非同期学習タスク (v4: 変化率＆Attention)
# ---------------------------------------------------------
def run_training_task(code: str):
    db = SessionLocal()
    try:
        update_status(code, {"status": "training", "progress": 5, "message": "株価データをロード中..."})
        query = text("SELECT date, close, volume FROM stock_prices WHERE code = :code ORDER BY date ASC")
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn, params={"code": code})
        
        # 既存DBへの補完ロジック
        if df.empty or len(df) < 200:
            ticker = yf.Ticker(f"{code}.T")
            hist = ticker.history(period="3y")
            if not hist.empty:
                hist = safe_complement_historical_data(code, ticker, hist)
                
            with engine.connect() as conn:
                existing = set(
                    pd.read_sql(
                        text("SELECT date FROM stock_prices WHERE code = :code"),
                        conn, params={"code": code}
                    )['date'].tolist()
                ) if not df.empty else set()
            
            records = _hist_to_records(code, hist)
            # 既に存在する日付をフィルタリング
            records = [r for r in records if r.date not in existing]
            if records:
                db.bulk_save_objects(records)
                db.commit()
            with engine.connect() as conn:
                df = pd.read_sql(query, con=conn, params={"code": code})

        update_status(code, {"status": "training", "progress": 15, "message": "特徴量とマクロ指標を計算中..."})
        
        # 変化率とS&P500を追加した特徴量生成
        df = compute_features(df)
        fm = extract_feature_matrix(df)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(fm)

        xs, ys = [], []
        # Targetは index 0 の `close_return` (変化率)
        for i in range(len(scaled) - SEQ_LENGTH - PREDICT_DAYS + 1):
            xs.append(scaled[i:(i + SEQ_LENGTH)])
            ys.append(scaled[i + SEQ_LENGTH : i + SEQ_LENGTH + PREDICT_DAYS, 0])
        X, y = np.array(xs), np.array(ys)

        sp = int(len(X) * 0.8)
        Xt, Xv = torch.tensor(X[:sp], dtype=torch.float32), torch.tensor(X[sp:], dtype=torch.float32)
        yt, yv = torch.tensor(y[:sp], dtype=torch.float32), torch.tensor(y[sp:], dtype=torch.float32)

        update_status(code, {"status": "training", "progress": 20, "message": f"Attention LSTM学習中... ({len(X)}サンプル)"})

        model = StockAttentionLSTM(input_size=FEATURE_COUNT, hidden_size=64, num_layers=2, dropout=0.3)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        EPOCHS, PATIENCE = 120, 15
        best_val = float('inf')
        wait, best_state = 0, None

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
                best_val, wait, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1

            if (ep + 1) % 5 == 0 or ep == 0:
                prog = 20 + int(((ep + 1) / EPOCHS) * 70)
                update_status(code, {"status": "training", "progress": min(prog, 90), "message": f"Epoch {ep+1} | Val Loss: {vl:.5f}"})

            if wait >= PATIENCE:
                break

        if best_state:
            model.load_state_dict(best_state)
        
        # セキュアなパスでモデル保存
        model_path = get_model_path(code)
        scaler_path = get_scaler_path(code)
        
        torch.save(model.state_dict(), model_path)

        # スケーラー保存 (v4)
        sdata = {
            "feature_names": ["close_return", "volume_norm", "sma5_dev", "rsi", "macd", "n225_return", "usdjpy_return", "sp500_return"],
            "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
            "input_size": FEATURE_COUNT, "model_version": 4, "val_mape": None
        }
        with open(scaler_path, "w") as f:
            json.dump(sdata, f)

        # 未来5日間予測を生成してステータスに格納
        update_status(code, {"status": "training", "progress": 95, "message": "未来5日間の予測を生成中..."})
        
        last_actual_price = float(df['close'].iloc[-1])
        seq = scaled[-SEQ_LENGTH:].copy()
        xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            preds_scaled_returns = model(xt).numpy().flatten()
            
        ret_min, ret_max = scaler.data_min_[0], scaler.data_max_[0]
        ret_range = ret_max - ret_min
        actual_returns = [(p * ret_range + ret_min) for p in preds_scaled_returns]
        
        predicted_prices = []
        current_price = last_actual_price
        for r in actual_returns:
            current_price = current_price * (1 + r)
            predicted_prices.append(round(current_price, 1))

        update_status(code, {
            "status": "success",
            "progress": 100,
            "prediction": predicted_prices[0],
            "predictions": predicted_prices,
            "mape": None,
            "message": "success"
        })
        logger.info(f"[{code}] v4 (Attention & Returns) 学習完了")
    except Exception as e:
        logger.error(f"Training error for {code}: {e}", exc_info=True)
        # セキュリティ: 内部エラーの詳細をクライアントに返さない
        update_status(code, {"status": "failed", "progress": 0, "message": "学習処理中にエラーが発生しました。ログを確認してください。"})
    finally:
        db.close()


@app.post("/api/train/{code}")
def train_stock_model(code: str, background_tasks: BackgroundTasks):
    code = validate_stock_code(code)
    
    # 同一銘柄の重複学習を防止
    if _is_training(code):
        return {"code": code, "status": "training", "message": "この銘柄はすでに学習中です。"}
    
    update_status(code, {"status": "training", "progress": 0, "message": "学習初期化中..."})
    background_tasks.add_task(run_training_task, code)
    return {"code": code, "status": "training"}


@app.get("/api/train/status/{code}")
def get_train_status(code: str):
    code = validate_stock_code(code)
    return get_status_safe(code)


# ---------------------------------------------------------
# 8. 企業情報 API
# ---------------------------------------------------------
@app.get("/api/info/{code}")
def get_stock_info(code: str):
    code = validate_stock_code(code)
    try:
        ticker = yf.Ticker(f"{code}.T")
        info = ticker.info
        if not info or not isinstance(info, dict):
            raise ValueError("Empty or invalid info returned")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching info for {code}: {e}")
        return {
            "code": code,
            "name": "不明",
            "marketCap": None,
            "trailingPE": None,
            "priceToBook": None,
            "dividendYield": None,
            "fiftyTwoWeekHigh": None,
            "fiftyTwoWeekLow": None,
            "previousClose": None
        }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)