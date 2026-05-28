# 株価分析ダッシュボード (kabukakasika)

FastAPI（バックエンド）と React（フロントエンド）を組み合わせた、リアルタイム株価分析ダッシュボードです。日本の主要銘柄の株価推移と出来高をグラフで可視化します。

## 🚀 プロジェクト構成

```text
kabukakasika/
├── backend/            # Python (FastAPI) サーバー
│   ├── .venv/          # Python 仮想環境
│   ├── db.py           # データベース共通設定
│   ├── validators.py   # 安全な入力値検証・パス生成
│   ├── models.py       # AI予測モデル（Attention LSTM）定義
│   ├── features.py     # 特徴量エンジニアリング共通処理
│   └── main.py         # APIサーバーメインルーチン
├── frontend/           # React (Vite) アプリケーション
│   ├── src/            # ソースコード (StockDashboard.tsx, types.ts 等)
│   └── package.json    # Frontend 依存関係
├── docs/               # 設計・運用ドキュメント
│   └── refactoring_and_security.md # セキュリティ対策およびリファクタリング報告書
└── README.md           # このファイル
```

## 🛠 セットアップと起動方法

### 1. バックエンド (Backend)
Python 3.12+ を使用します。

1. `backend` ディレクトリへ移動:
   ```powershell
   cd backend
   ```
2. サーバーの起動:
   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```
   ※ サーバーは `http://localhost:8000` で起動します。

### 2. フロントエンド (Frontend)
Node.js が必要です。

1. `frontend` ディレクトリへ移動:
   ```powershell
   cd frontend
   ```
2. 依存ライブラリのインストール (初回のみ):
   ```powershell
   npm install
   ```
3. 開発サーバーの起動:
   ```powershell
   npm run dev
   ```
   ※ 通常 `http://localhost:5173` でブラウザからアクセス可能です。

## 📈 主な機能
- **銘柄選択**: トヨタ、ソニー、任天堂などの主要銘柄をドロップダウンで選択可能。
- **株価チャート**: 過去3ヶ月間の終値推移を LineChart で表示。
- **出来高チャート**: 出来高の推移を BarChart で表示。
- **自動更新**: 銘柄を変更すると即座にバックエンドから最新データを取得します。

## 📚 使用技術
- **Backend**: Python, FastAPI, yfinance, Pandas, Uvicorn
- **Frontend**: React (TypeScript), Vite, Recharts (グラフ描画)

## 🔧 データ堅牢化とトラブルシューティング
株価データ取得時の yfinance 由来のデータ欠損（最新営業日の終値 NaN 問題）に対して、自動で `period='1d'` からデータを再取得・補完する自己修復ロジックを実装しています。
詳細な仕様や手動修復手順については、[yfinance最新データ欠損対策の仕様ドキュメント](file:///c:/Users/nabe4/kabukakasika/backend/yfinance_nan_handling.md) を参照してください。

## 🛡️ セキュリティ対策とリファクタリング
システムの堅牢性と保守性を向上させるため、SQLインジェクション対策、パストラバーサル防御、入力値の厳密なバリデーション、およびバックエンドモジュールの機能分割を実施しました。
具体的な変更内容と設計方針については、[セキュリティ対策およびリファクタリング実施報告書](file:///c:/Users/nabe4/kabukakasika/docs/refactoring_and_security.md) を参照してください。
