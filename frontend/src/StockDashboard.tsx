import { useState, useEffect } from 'react';
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

import StockSearchBox from './StockSearchBox';


export default function StockDashboard() {
  const [selectedCode, setSelectedCode] = useState<string>("7203"); // 初期値はトヨタ自動車(7203)
  const [stockData, setStockData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // 指標の表示/非表示を管理するステート
  const [showSMA5, setShowSMA5] = useState<boolean>(true);
  const [showSMA25, setShowSMA25] = useState<boolean>(true);
  const [showBB, setShowBB] = useState<boolean>(true);

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
      <h1 style={{ borderBottom: '2px solid #eee', paddingBottom: '10px' }}>株価分析ダッシュボード (v2.0 - Search Enabled)</h1>
      
      {/* 操作パネルエリア */}
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', margin: '20px 0', padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <label style={{ marginRight: '10px', fontWeight: 'bold' }}>銘柄検索:</label>
          
          {/* ここにオートコンプリートを配置 */}
          <StockSearchBox 
            initialCode={selectedCode}
            onSelect={(code) => setSelectedCode(code)} 
          />
          
        </div>

        {/* 指標のトグルスイッチ */}
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center', borderLeft: '1px solid #ddd', paddingLeft: '20px' }}>
          <span style={{ fontWeight: 'bold' }}>表示指標:</span>
          <label style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={showSMA5} onChange={(e) => setShowSMA5(e.target.checked)} /> 5日移動平均
          </label>
          <label style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={showSMA25} onChange={(e) => setShowSMA25(e.target.checked)} /> 25日移動平均
          </label>
          <label style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={showBB} onChange={(e) => setShowBB(e.target.checked)} /> ボリンジャーバンド
          </label>
        </div>
      </div>

      {/* ローディング・エラー表示 */}
      {loading && <p>データを取得中...</p>}
      {error && <p style={{ color: 'red' }}>エラー: {error}</p>}

      {/* チャート表示エリア */}
      {!loading && !error && stockData.length > 0 && (
        <div>
          <h2 style={{ fontSize: '18px', marginTop: '30px' }}>株価推移・テクニカル指標</h2>
          <div style={{ height: '500px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stockData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{fontSize: 12}} />
                <YAxis domain={['auto', 'auto']} tick={{fontSize: 12}} />
                <Tooltip />
                
                {/* 終値 (ベースライン) */}
                <Line type="monotone" dataKey="close" stroke="#333333" strokeWidth={2} dot={false} name="終値" />
                
                {/* テクニカル指標 (チェックボックスに連動) */}
                {showSMA5 && (
                  <Line type="monotone" dataKey="sma5" stroke="#ff7300" strokeWidth={1.5} dot={false} name="5日SMA" />
                )}
                {showSMA25 && (
                  <Line type="monotone" dataKey="sma25" stroke="#387908" strokeWidth={1.5} dot={false} name="25日SMA" />
                )}
                {showBB && (
                  <Line type="monotone" dataKey="bbUpper" stroke="#cc00ff" strokeWidth={2} strokeDasharray="10 5" dot={false} name="+2σ" />
                )}
                {showBB && (
                  <Line type="monotone" dataKey="bbLower" stroke="#cc00ff" strokeWidth={2} strokeDasharray="10 5" dot={false} name="-2σ" />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h2 style={{ fontSize: '18px', marginTop: '30px' }}>出来高</h2>
          <div style={{ height: '200px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stockData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{fontSize: 12}} />
                <YAxis tick={{fontSize: 12}} />
                <Tooltip />
                <Bar dataKey="volume" fill="#82ca9d" name="出来高" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
