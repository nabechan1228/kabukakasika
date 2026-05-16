import pandas as pd
import json
import os

# JPX（日本取引所グループ）の公式公開データURL（毎月更新）
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

def generate_stock_master():
    print("JPXの公式サイトから最新の上場銘柄データをダウンロードしています...")
    
    try:
        # Excelデータを直接Pandasで読み込む
        df = pd.read_excel(JPX_URL)
        
        # 必要なカラム（コードと銘柄名）だけを抽出
        df = df[['コード', '銘柄名']]
        
        # データの整形とJSON用のリスト作成
        master_data = []
        for index, row in df.iterrows():
            code = str(row['コード']).strip()
            name = str(row['銘柄名']).strip()
            
            # 空データや異常値を弾く
            if len(code) == 4:
                master_data.append({
                    "code": code,
                    "name": name
                })
        
        # フロントエンドのフォルダ（srcディレクトリ）へ保存する想定のパス
        # ※Reactプロジェクトの src フォルダのパスに合わせて変更してください
        output_path = "../frontend/src/stock_master.json" 
        
        # 保存先ディレクトリが存在しない場合はカレントディレクトリに保存
        if not os.path.exists(os.path.dirname(output_path)):
            output_path = "stock_master.json"
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, ensure_ascii=False, indent=2)
            
        print(f"成功: {len(master_data)}件の銘柄データを {output_path} に保存しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    generate_stock_master()
