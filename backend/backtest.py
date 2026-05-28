"""
バックテストスクリプト: 過去の株価データに対してモデルの予測精度を評価する。

使い方:
    python backtest.py
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text

# 共通モジュールからインポート
from db import engine
from models import (
    StockAttentionLSTM, FEATURE_COUNT, SEQ_LENGTH,
    restore_scaler, load_model
)
from features import compute_features, extract_feature_matrix
from validators import get_model_path, get_scaler_path


def run_backtest(code: str, test_days: int = 30):
    print(f"[{code}] バックテストを開始します（テスト期間: 過去 {test_days} 営業日）...")

    model_path = get_model_path(code)
    scaler_path = get_scaler_path(code)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("モデルまたはスケーラーが見つかりません。先に学習を行ってください。")
        return

    # 1. スケーラーとモデルの復元（共通ヘルパー使用）
    scaler, si = restore_scaler(scaler_path)
    model = load_model(model_path, input_size=si.get("input_size", FEATURE_COUNT))

    ret_min, ret_max = scaler.data_min_[0], scaler.data_max_[0]
    ret_range = ret_max - ret_min

    # 2. データの準備（テスト期間＋シーケンス長＋特徴量計算の余白分を取得）
    # 【修正】SQLインジェクション対策: パラメータバインドを使用
    limit = test_days + SEQ_LENGTH + 60
    query = text(
        "SELECT date, close, volume FROM stock_prices "
        "WHERE code = :code ORDER BY date DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, con=conn, params={"code": code, "limit": limit})
    
    df = df.iloc[::-1].reset_index(drop=True)
    df = compute_features(df)
    
    # NaNを落とす前の全体のインデックスと日付を保持
    valid_df = df.dropna().reset_index(drop=True)
    fm = extract_feature_matrix(valid_df)
    scaled_fm = scaler.transform(fm)

    # 3. バックテストの実行
    results = []
    
    # valid_df の後ろから test_days 分をテスト対象とする
    start_idx = len(valid_df) - test_days
    
    for i in range(start_idx, len(valid_df)):
        current_date = valid_df.loc[i, 'date']
        actual_close = valid_df.loc[i, 'close']
        
        # 予測に使用するのは「前日までの60日間」
        seq_start = i - SEQ_LENGTH
        seq_end = i
        
        if seq_start < 0:
            continue  # データ不足
            
        seq = scaled_fm[seq_start:seq_end]
        xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        
        # 前日の終値（復元用基準値）
        prev_actual_close = valid_df.loc[i-1, 'close']

        # 推論
        with torch.no_grad():
            preds_scaled = model(xt).numpy().flatten()
            
        # 翌日（T+1）の変化率を取り出し、実価格に変換
        pred_return = preds_scaled[0] * ret_range + ret_min
        pred_close = prev_actual_close * (1 + pred_return)
        
        # 前日からの方向（Up=1, Down=-1）
        actual_dir = 1 if actual_close > prev_actual_close else -1
        pred_dir = 1 if pred_close > prev_actual_close else -1
        correct_dir = 1 if actual_dir == pred_dir else 0

        results.append({
            'date': current_date,
            'actual': actual_close,
            'predicted': pred_close,
            'correct_dir': correct_dir
        })

    # 4. 評価指標の計算と結果表示
    res_df = pd.DataFrame(results)
    
    mae = np.mean(np.abs(res_df['actual'] - res_df['predicted']))
    rmse = np.sqrt(np.mean((res_df['actual'] - res_df['predicted'])**2))
    dir_acc = res_df['correct_dir'].mean() * 100

    print("\n" + "="*40)
    print(f"バックテスト結果: {code} (過去 {len(res_df)} 営業日)")
    print("="*40)
    print(f"方向勝率 (Up/Down): {dir_acc:.1f}%")
    print(f"MAE (平均絶対誤差): {mae:.1f} 円")
    print(f"RMSE (二乗平均平方根誤差): {rmse:.1f} 円")
    print("="*40)

    # 5. グラフの描画
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['date'], res_df['actual'], label='Actual Price', color='black', marker='o', markersize=4)
    plt.plot(res_df['date'], res_df['predicted'], label='Predicted Price (1-day ahead)', color='red', linestyle='--', marker='x', markersize=4)
    
    plt.title(f"Backtest Results for {code}")
    plt.xlabel('Date')
    plt.ylabel('Price (JPY)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 例としてトヨタ（7203）の過去30営業日をテスト
    run_backtest("7203", test_days=30)
