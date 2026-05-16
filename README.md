# 株価分析ダッシュボード (kabukakasika)

FastAPI（バックエンド）と React（フロントエンド）を組み合わせた、リアルタイム株価分析ダッシュボードです。日本の主要銘柄の株価推移と出来高をグラフで可視化します。

## 🚀 プロジェクト構成

```text
kabukakasika/
├── backend/            # Python (FastAPI) サーバー
│   ├── .venv/          # Python 仮想環境
│   └── main.py         # APIサーバー実装 (yfinanceを使用)
├── frontend/           # React (Vite) アプリケーション
│   ├── src/            # ソースコード (StockDashboard.tsx 等)
│   └── package.json    # Frontend 依存関係
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
