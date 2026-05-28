"""
特徴量エンジニアリングとマクロ経済指標キャッシュ。

株価データに対して技術的指標（RSI, MACD, SMA等）とマクロ経済指標
（日経225, USD/JPY, S&P500）を結合した特徴量行列を生成する。
"""
import datetime as dt
import logging
import threading

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("kabukakasika")

# ---------------------------------------------------------
# マクロ指標のグローバルキャッシュ
# ---------------------------------------------------------
_macro_cache_lock = threading.Lock()
_macro_cache = {"last_fetched": None, "data": None}


def get_macro_data() -> pd.DataFrame:
    """
    マクロ経済指標（日経225、USD/JPY、S&P500）の日次変化率を取得する。

    1時間のキャッシュ機構があり、同一プロセス内での重複ダウンロードを防止する。
    """
    global _macro_cache
    now = dt.datetime.now()

    # 1. ロック内で更新が必要かどうかのみ判定する
    need_fetch = False
    with _macro_cache_lock:
        if _macro_cache["data"] is None or _macro_cache["last_fetched"] is None:
            need_fetch = True
        elif (now - _macro_cache["last_fetched"]).total_seconds() > 3600:  # 1時間キャッシュ
            need_fetch = True

    # 2. ロックの外側で重い通信（ダウンロード）を行う
    if need_fetch:
        try:
            # 常に過去3年分のマクロデータを一括取得（学習と推論の全範囲をカバー）
            cache_start = (now - dt.timedelta(days=3 * 365)).strftime('%Y-%m-%d')
            logger.info(f"マクロ指標キャッシュを更新中... (開始日: {cache_start})")

            def download_macro(ticker_symbol: str) -> pd.Series:
                df = yf.download(ticker_symbol, start=cache_start, progress=False)
                if df.empty:
                    raise ValueError(f"データが空です: {ticker_symbol}")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Close' not in df.columns:
                    raise ValueError(f"Closeカラムが見つかりません: {ticker_symbol}")
                series = df['Close'].squeeze()
                if isinstance(series.index, pd.DatetimeIndex):
                    series.index = series.index.tz_localize(None)
                return series

            n225 = download_macro("^N225")
            fx = download_macro("USDJPY=X")
            sp500 = download_macro("^GSPC")

            macro_df = pd.DataFrame({
                'n225_return': n225.pct_change(),
                'usdjpy_return': fx.pct_change(),
                'sp500_return': sp500.pct_change()
            }).reset_index()

            date_col = 'Date' if 'Date' in macro_df.columns else macro_df.columns[0]
            macro_df['date'] = pd.to_datetime(macro_df[date_col]).dt.strftime('%Y-%m-%d')

            new_data = macro_df[['date', 'n225_return', 'usdjpy_return', 'sp500_return']]

            # 3. 取得完了後、再度ロックを獲得してキャッシュを更新する
            with _macro_cache_lock:
                _macro_cache["data"] = new_data
                _macro_cache["last_fetched"] = now
            logger.info("マクロ指標キャッシュの更新が完了しました。")
        except Exception as e:
            logger.error(f"マクロ指標のインターネット取得に失敗しました。: {e}", exc_info=True)
            with _macro_cache_lock:
                if _macro_cache["data"] is None:
                    # 初回取得失敗時のフォールバック
                    _macro_cache["data"] = pd.DataFrame(
                        columns=['date', 'n225_return', 'usdjpy_return', 'sp500_return']
                    )
                    _macro_cache["last_fetched"] = now

    # 4. キャッシュされたデータを返す（読み取り時もロックを取得）
    with _macro_cache_lock:
        return _macro_cache["data"].copy()


# ---------------------------------------------------------
# 特徴量エンジニアリング
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    'close_return', 'volume_norm', 'sma5_dev', 'rsi', 'macd',
    'n225_return', 'usdjpy_return', 'sp500_return'
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    株価DataFrameにテクニカル指標とマクロ経済指標を追加する。

    Args:
        df: 'date', 'close', 'volume' カラムを含むDataFrame

    Returns:
        特徴量カラムが追加されたDataFrame
    """
    # 前日からの変化率をメイン特徴量にする
    df['close_return'] = df['close'].pct_change()

    sma5 = df['close'].rolling(window=5).mean()
    df['sma5_dev'] = ((df['close'] - sma5) / sma5) * 100

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss_s
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = (
        df['close'].ewm(span=12, adjust=False).mean()
        - df['close'].ewm(span=26, adjust=False).mean()
    )
    vol_max = df['volume'].max()
    df['volume_norm'] = df['volume'] / vol_max if vol_max > 0 else 0

    try:
        # キャッシュからマクロ指標をマージ
        macro_df = get_macro_data()
        df = pd.merge(df, macro_df, on='date', how='left')
    except Exception as e:
        logger.error(f"マクロ指標の結合に失敗しました: {e}")
        df['n225_return'] = 0
        df['usdjpy_return'] = 0
        df['sp500_return'] = 0

    df.fillna(0, inplace=True)  # NaNをゼロで埋める
    return df


def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    DataFrameから学習/推論用の特徴量行列(numpy配列)を抽出する。

    Args:
        df: compute_features() 適用済みのDataFrame

    Returns:
        shape (n_samples, 8) の numpy 配列
    """
    return df[FEATURE_COLUMNS].values
