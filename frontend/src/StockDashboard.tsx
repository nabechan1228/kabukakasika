import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';

// 有名銘柄リスト（マスターデータ）
const STOCK_LIST = [
  { name: "トヨタ自動車", code: "7203" },
  { name: "ソニーグループ", code: "6758" },
  { name: "ソフトバンクグループ", code: "9984" },
  { name: "任天堂", code: "7974" },
  { name: "ファーストリテイリング", code: "9983" }
];

export default function StockDashboard() {
  const [selectedCode, setSelectedCode] = useState<string>(STOCK_LIST[0].code);
  const [stockData, setStockData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // 銘柄が変更されたらAPIからデータを取得
  useEffect(() => {
    const fetchStockData = async () => {
      setLoading(true);
      setError(null);
      try {
        // FastAPIのバックエンドへリクエスト
        const response = await fetch(`http://localhost:8000/api/stock/${selectedCode}`);
        if (!response.ok) {
          throw new Error('データの取得に失敗しました');
        }
        const data = await response.json();
        setStockData(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
  }, [selectedCode]);

  return (
    <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto', fontFamily: 'sans-serif' }}>
      <h1 style={{ borderBottom: '2px solid #eee', paddingBottom: '10px' }}>株価分析ダッシュボード</h1>
      
      {/* 銘柄選択 UI */}
      <div style={{ margin: '20px 0' }}>
        <label htmlFor="stock-select" style={{ marginRight: '10px', fontWeight: 'bold' }}>
          銘柄を選択:
        </label>
        <select 
          id="stock-select"
          value={selectedCode} 
          onChange={(e) => setSelectedCode(e.target.value)}
          style={{ padding: '8px', fontSize: '16px', borderRadius: '4px' }}
        >
          {STOCK_LIST.map((stock) => (
            <option key={stock.code} value={stock.code}>
              {stock.name} ({stock.code})
            </option>
          ))}
        </select>
      </div>

      {/* ローディング・エラー表示 */}
      {loading && <p>データを取得中...</p>}
      {error && <p style={{ color: 'red' }}>エラー: {error}</p>}

      {/* チャート表示エリア */}
      {!loading && !error && stockData.length > 0 && (
        <div>
          <h2 style={{ fontSize: '18px', marginTop: '30px' }}>株価推移（終値）</h2>
          <div style={{ height: '400px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stockData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{fontSize: 12}} />
                {/* domain設定でグラフの上下に余白を持たせ、変化を分かりやすくする */}
                <YAxis domain={['auto', 'auto']} tick={{fontSize: 12}} />
                <Tooltip />
                <Line type="monotone" dataKey="close" stroke="#8884d8" strokeWidth={2} dot={false} name="終値(円)" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h2 style={{ fontSize: '18px', marginTop: '30px' }}>出来高</h2>
          <div style={{ height: '200px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stockData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{fontSize: 12}} />
                <YAxis tick={{fontSize: 12}} />
                <Tooltip />
                <Bar dataKey="volume" fill="#82ca9d" name="出来高(株)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
