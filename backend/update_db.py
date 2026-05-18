import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# データベースの接続設定（main.pyと同じ）
DATABASE_URL = "sqlite:///./stock_data.db"
engine = create_engine(DATABASE_URL)

def update_stock_data():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 株価データのバッチ更新を開始します...")

    try:
        # 1. データベースに登録されている銘柄コードを重複なしで取得
        from sqlalchemy import text as sa_text
        query_codes = sa_text("SELECT DISTINCT code FROM stock_prices")
        with engine.connect() as conn:
            codes_df = pd.read_sql(query_codes, con=conn)
        
        if codes_df.empty:
            print("データベースに銘柄が登録されていません。まずは画面から銘柄を選択して初回データを取得してください。")
            return

        codes = codes_df['code'].tolist()

        # 各銘柄ごとに差分更新処理を行う
        for code in codes:
            print(f"\n--- 銘柄 [{code}] の更新処理 ---")
            
            # 2. データベース内にあるこの銘柄の「最も新しい日付」を取得
            from sqlalchemy import text
            query_max_date = text("SELECT MAX(date) as max_date FROM stock_prices WHERE code = :code")
            with engine.connect() as conn:
                max_date_df = pd.read_sql(query_max_date, con=conn, params={"code": code})
            last_date_str = max_date_df.iloc[0]['max_date']

            if not last_date_str:
                print(f"[{code}] データがありません。スキップします。")
                continue

            # 文字列の日付を日付型に変換し、取得開始日（保存されている最新日の翌日）を計算
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')

            # 今日よりも未来の日付になっていれば、すでに最新
            if start_date > today:
                print(f"[{code}] すでに最新のデータ（{last_date_str}）が保存されています。")
                continue

            print(f"[{code}] {start_date} 以降の新しいデータを取得します...")
            
            # 3. yfinanceから差分データを取得
            ticker_symbol = f"{code}.T"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=start_date)

            if hist.empty:
                print(f"[{code}] 新しいデータはありませんでした。")
                continue

            # 4. データベースのテーブル定義に合わせてデータを整形
            records = []
            for index, row in hist.iterrows():
                date_str = index.strftime('%Y-%m-%d')
                
                # yfinanceの仕様上、指定した開始日以前のデータが混ざることがあるため重複をブロック
                if date_str <= last_date_str:
                    continue

                records.append({
                    "code": code,
                    "date": date_str,
                    "open": round(row['Open'], 1),
                    "high": round(row['High'], 1),
                    "low": round(row['Low'], 1),
                    "close": round(row['Close'], 1),
                    "volume": int(row['Volume'])
                })

            if not records:
                print(f"[{code}] 新規追加するデータはありませんでした。")
                continue

            # 5. Pandasの機能を使ってデータベースに一括追記（append）
            insert_df = pd.DataFrame(records)
            insert_df.to_sql('stock_prices', con=engine, if_exists='append', index=False)
            
            print(f"[{code}] {len(records)} 件のデータを新たに追加しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] バッチ更新が完了しました。")

if __name__ == "__main__":
    update_stock_data()
