"""
Attention付きLSTMモデル定義とスケーラー復元ユーティリティ。

推論(predict)、学習(train)、バックテスト(backtest)で
共通利用されるモデルアーキテクチャとスケーラー復元ロジックを集約。
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# モデルハイパーパラメータ定数
# ---------------------------------------------------------
FEATURE_COUNT = 8  # [close_return, volume_norm, sma5_dev, rsi, macd, n225_return, usdjpy_return, sp500_return]
SEQ_LENGTH = 60
PREDICT_DAYS = 5


# ---------------------------------------------------------
# Attention付き LSTMモデル (v4: 8次元マクロ・変化率対応)
# ---------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Attentionの重みを計算するための層
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_length, hidden_size)
        attn_weights = F.softmax(self.attention(lstm_outputs), dim=1)  # (batch, seq, 1)
        # 重みと出力を掛け合わせてコンテキストベクトルを生成
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)  # (batch, hidden_size)
        return context_vector, attn_weights


class StockAttentionLSTM(nn.Module):
    def __init__(self, input_size=FEATURE_COUNT, hidden_size=64, num_layers=2, dropout=0.3):
        super(StockAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, PREDICT_DAYS)  # 未来5日分の「変化率」を出力

    def forward(self, x):
        out, _ = self.lstm(x)
        # 最後の出力だけを使うのではなく、Attentionで全期間から重要な情報を抽出
        context, attn_weights = self.attention(out)
        context = self.dropout(context)
        predictions = self.fc(context)
        return predictions


# ---------------------------------------------------------
# スケーラー復元ヘルパー
# ---------------------------------------------------------
def restore_scaler(scaler_path: str) -> tuple[MinMaxScaler, dict]:
    """
    保存済みスケーラーJSONからMinMaxScalerを復元する。

    Args:
        scaler_path: スケーラーJSONファイルのパス

    Returns:
        (scaler, scaler_info) のタプル。
        scaler_info は元のJSONデータ（input_size, val_mape等を含む）。

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 次元数が不一致の場合
    """
    with open(scaler_path, "r") as f:
        si = json.load(f)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = np.array(si["data_min"])
    scaler.data_max_ = np.array(si["data_max"])

    # 旧バージョンのスケーラーとの次元不一致を検出
    if len(scaler.data_min_) != FEATURE_COUNT:
        raise ValueError(
            f"スケーラーの次元数 ({len(scaler.data_min_)}) がモデルの特徴量数 "
            f"({FEATURE_COUNT}) と一致しません。再学習を行ってください。"
        )

    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = 1.0 / np.where(scaler.data_range_ == 0, 1, scaler.data_range_)
    scaler.min_ = 0.0 - scaler.data_min_ * scaler.scale_

    return scaler, si


def load_model(model_path: str, input_size: int = FEATURE_COUNT) -> StockAttentionLSTM:
    """
    保存済みモデルをロードして評価モードで返す。

    Args:
        model_path: モデルファイルのパス
        input_size: 入力特徴量の次元数

    Returns:
        評価モードに設定されたStockAttentionLSTMインスタンス
    """
    model = StockAttentionLSTM(input_size=input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model
