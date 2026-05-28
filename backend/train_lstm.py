"""
LSTMモデルのスタンドアロン学習スクリプト（v1: 単純LSTMの基本版）。

注意: このスクリプトは初期開発時のシンプルなLSTMモデルを学習するものです。
現在のAPIで使用されるAttention付きLSTM (v4) の学習は main.py の
/api/train/{code} エンドポイントから実行してください。

使い方:
    python train_lstm.py
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
from sqlalchemy import text

# 共通モジュールからインポート
from db import engine

# ---------------------------------------------
# 1. データ準備と前処理
# ---------------------------------------------
# ターゲット銘柄（例としてトヨタを指定。将来的にはループ処理で全銘柄対応可能）
TARGET_CODE = "7203"

print(f"[{TARGET_CODE}] のデータを読み込んでいます...")
query = text("SELECT date, close FROM stock_prices WHERE code = :code ORDER BY date ASC")
with engine.connect() as conn:
    df = pd.read_sql(query, con=conn, params={"code": TARGET_CODE})

if len(df) < 100:
    raise ValueError("学習に十分なデータがありません。")

# 終値のみを抽出（1次元配列を2次元に変換）
prices = df['close'].values.reshape(-1, 1)

# ニューラルネットは 0〜1 の値で最も効率よく学習するため、正規化します
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# スケーラーを保存（FastAPIでの推論時、予測値を元の株価のスケールに戻すために必須です。安全なJSON形式で保存します）
scaler_data = {
    "data_min": scaler.data_min_.tolist(),
    "data_max": scaler.data_max_.tolist()
}
with open(f"scaler_{TARGET_CODE}.json", "w") as f:
    json.dump(scaler_data, f)

# ---------------------------------------------
# 2. シーケンスデータの作成（過去60日 -> 翌日）
# ---------------------------------------------
SEQ_LENGTH = 60

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_prices, SEQ_LENGTH)

# PyTorchのテンソルに変換
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# ---------------------------------------------
# 3. LSTMモデルの定義 (PyTorch)
# ---------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTMレイヤー: batch_first=True で (バッチサイズ, シーケンス長, 特徴量) の入力を受け付ける
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 最終出力を株価（1つの値）にする全結合層
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTMからの出力と、隠れ状態を取得
        out, _ = self.lstm(x)
        # シーケンスの最後の日（-1）の出力を全結合層に渡す
        predictions = self.linear(out[:, -1, :])
        return predictions

model = StockLSTM()

# ---------------------------------------------
# 4. 学習ループ
# ---------------------------------------------
criterion = nn.MSELoss()  # 回帰タスクなので平均二乗誤差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
print("学習を開始します...")

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # 順伝播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # 逆伝播と最適化
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

# ---------------------------------------------
# 5. モデルの保存
# ---------------------------------------------
model_path = f"lstm_model_{TARGET_CODE}.pth"
torch.save(model.state_dict(), model_path)
print(f"学習完了！モデルを {model_path} に保存しました。")
