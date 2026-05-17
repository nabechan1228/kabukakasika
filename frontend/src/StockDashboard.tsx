import { useState, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  ComposedChart,
  Cell
} from 'recharts';

import StockSearchBox from './StockSearchBox';


export default function StockDashboard() {
  const [selectedCode, setSelectedCode] = useState<string>("7203");
  const [stockData, setStockData] = useState<any[]>([]);
  const [stockInfo, setStockInfo] = useState<any>(null); // 追加: 企業情報
  const [watchlist, setWatchlist] = useState<{code: string, name: string}[]>([]); // 追加: ウォッチリスト
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [training, setTraining] = useState<boolean>(false);
  
  const [chartType, setChartType] = useState<'line' | 'candle'>('candle');

  // 指標トグル
  const [showSMA5, setShowSMA5] = useState<boolean>(true);
  const [showSMA25, setShowSMA25] = useState<boolean>(true);
  const [showBB, setShowBB] = useState<boolean>(true);
  const [showRSI, setShowRSI] = useState<boolean>(true);   // 追加: RSI
  const [showMACD, setShowMACD] = useState<boolean>(true); // 追加: MACD

  // ウォッチリストのロード
  useEffect(() => {
    const saved = localStorage.getItem('kabukakasika_watchlist');
    if (saved) {
      setWatchlist(JSON.parse(saved));
    }
  }, []);

  const handleToggleWatchlist = () => {
    // 企業名がわからない場合はスキップ（安全対策）
    const name = stockInfo?.name || "銘柄";
    const exists = watchlist.find(w => w.code === selectedCode);
    let updated;
    if (exists) {
      updated = watchlist.filter(w => w.code !== selectedCode);
    } else {
      updated = [...watchlist, { code: selectedCode, name }];
    }
    setWatchlist(updated);
    localStorage.setItem('kabukakasika_watchlist', JSON.stringify(updated));
  };

  const chartData = useMemo(() => {
    return stockData.map(item => ({
      ...item,
      lowHigh: [item.low, item.high],
      openClose: [Math.min(item.open, item.close), Math.max(item.open, item.close)],
      isUp: item.close >= item.open
    }));
  }, [stockData]);

  useEffect(() => {
    const fetchStockData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://localhost:8000/api/stock/${selectedCode}`);
        if (!response.ok) throw new Error('データの取得に失敗しました');
        const data = await response.json();
        setStockData(data);

        // 企業情報の取得
        try {
          const infoRes = await fetch(`http://localhost:8000/api/info/${selectedCode}`);
          if (infoRes.ok) setStockInfo(await infoRes.json());
          else setStockInfo(null);
        } catch (e) {
          setStockInfo(null);
        }

        // AI予測の取得
        try {
          const predResponse = await fetch(`http://localhost:8000/api/predict/${selectedCode}`);
          if (predResponse.ok) {
            const predData = await predResponse.json();
            setPrediction(predData.prediction);
          } else setPrediction(null);
        } catch (e) {
          setPrediction(null);
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchStockData();
  }, [selectedCode]);

  const handleTrainModel = async () => {
    setTraining(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/api/train/${selectedCode}`, { method: 'POST' });
      if (!response.ok) throw new Error('AIの学習に失敗しました');
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  };

  const isWatched = watchlist.some(w => w.code === selectedCode);

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* トップヘッダー ＆ ウォッチリスト */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '10px', borderBottom: '1px solid var(--border-color)' }}>
        <h1 style={{ margin: '0 20px 0 0', fontSize: '24px', background: '-webkit-linear-gradient(45deg, #06b6d4, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', whiteSpace: 'nowrap' }}>
          PRO-TRADER
        </h1>
        
        {/* ウォッチリスト（お気に入り） */}
        <div style={{ display: 'flex', gap: '10px', overflowX: 'auto', padding: '5px', flexGrow: 1 }}>
          {watchlist.map(item => (
            <div 
              key={item.code}
              onClick={() => setSelectedCode(item.code)}
              style={{
                background: selectedCode === item.code ? 'rgba(6, 182, 212, 0.2)' : 'var(--panel-bg)',
                border: selectedCode === item.code ? '1px solid var(--accent-cyan)' : '1px solid var(--border-color)',
                padding: '6px 12px',
                borderRadius: '20px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: 'bold',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s',
                boxShadow: selectedCode === item.code ? '0 0 10px rgba(6, 182, 212, 0.3)' : 'none'
              }}
            >
              ⭐ {item.name} ({item.code})
            </div>
          ))}
        </div>
      </div>
      
      {/* 操作パネルエリア */}
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', margin: '0 0 20px 0', padding: '15px', backgroundColor: 'var(--panel-bg)', borderRadius: '12px', border: '1px solid var(--border-color)', backdropFilter: 'blur(10px)' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <label style={{ marginRight: '10px', fontWeight: 'bold', color: 'var(--text-muted)' }}>検索:</label>
          <StockSearchBox initialCode={selectedCode} onSelect={setSelectedCode} />
          
          <button 
            onClick={handleToggleWatchlist}
            style={{ 
              marginLeft: '15px', background: 'transparent', border: '1px solid var(--border-color)', 
              color: isWatched ? '#fbbf24' : 'var(--text-muted)', fontSize: '20px', 
              cursor: 'pointer', padding: '5px 10px', borderRadius: '8px',
              transition: 'all 0.2s', backgroundColor: 'rgba(255,255,255,0.05)'
            }}
            title="ウォッチリストに追加/削除"
          >
            {isWatched ? '★' : '☆'}
          </button>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', borderLeft: '1px solid var(--border-color)', paddingLeft: '20px' }}>
          <span style={{ fontWeight: 'bold', color: 'var(--text-muted)', fontSize: '13px' }}>形式:</span>
          <button onClick={() => setChartType('candle')} style={{ padding: '6px 12px', borderRadius: '6px', border: 'none', background: chartType === 'candle' ? 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)' : 'rgba(255,255,255,0.1)', color: '#fff', cursor: 'pointer', fontSize: '12px' }}>🕯️ ローソク足</button>
          <button onClick={() => setChartType('line')} style={{ padding: '6px 12px', borderRadius: '6px', border: 'none', background: chartType === 'line' ? 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)' : 'rgba(255,255,255,0.1)', color: '#fff', cursor: 'pointer', fontSize: '12px' }}>📈 折れ線</button>
        </div>

        <div style={{ display: 'flex', gap: '15px', alignItems: 'center', borderLeft: '1px solid var(--border-color)', paddingLeft: '20px', fontSize: '13px' }}>
          <span style={{ fontWeight: 'bold', color: 'var(--text-muted)' }}>指標:</span>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}><input type="checkbox" checked={showSMA5} onChange={(e) => setShowSMA5(e.target.checked)} /> 5日SMA</label>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}><input type="checkbox" checked={showSMA25} onChange={(e) => setShowSMA25(e.target.checked)} /> 25日SMA</label>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}><input type="checkbox" checked={showBB} onChange={(e) => setShowBB(e.target.checked)} /> ボリンジャー</label>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}><input type="checkbox" checked={showRSI} onChange={(e) => setShowRSI(e.target.checked)} /> RSI</label>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}><input type="checkbox" checked={showMACD} onChange={(e) => setShowMACD(e.target.checked)} /> MACD</label>
        </div>
      </div>

      {loading && <p style={{ color: 'var(--accent-cyan)' }}>データを取得中...</p>}
      {error && <p style={{ color: 'var(--up-color)' }}>エラー: {error}</p>}

      {!loading && !error && stockData.length > 0 && (
        <div style={{ display: 'flex', gap: '20px', flexDirection: 'row', flexWrap: 'wrap' }}>
          
          {/* 左側：チャートエリア (70%) */}
          <div style={{ flex: '1 1 65%', display: 'flex', flexDirection: 'column', gap: '15px', minWidth: '600px' }}>
            
            {/* メインチャート */}
            <div style={{ backgroundColor: 'var(--panel-bg)', padding: '15px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h2 style={{ fontSize: '16px', margin: 0, color: 'var(--text-muted)' }}>株価推移・テクニカル指標</h2>
                
                {prediction !== null ? (
                  <div style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)', padding: '6px 14px', borderRadius: '20px', color: '#fff', fontWeight: 'bold', fontSize: '13px', boxShadow: '0 0 10px rgba(236, 72, 153, 0.4)' }}>
                    🤖 AI予測 (翌日終値): {Math.round(prediction).toLocaleString()} 円
                  </div>
                ) : training ? (
                  <div style={{ background: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)', padding: '6px 14px', borderRadius: '20px', color: '#fff', fontWeight: 'bold', fontSize: '13px', opacity: 0.8 }}>
                    ⚡ AI学習中... (約5秒)
                  </div>
                ) : (
                  <button onClick={handleTrainModel} style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', border: 'none', padding: '6px 14px', borderRadius: '20px', color: '#fff', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px', boxShadow: '0 0 10px rgba(16, 185, 129, 0.4)' }}>
                    🤖 AI予測: 未学習 [学習開始]
                  </button>
                )}
              </div>

              <div style={{ height: '400px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} barGap="-100%" margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" tick={{fontSize: 11, fill: '#94a3b8'}} stroke="rgba(255,255,255,0.1)" />
                    <YAxis domain={['auto', 'auto']} tick={{fontSize: 11, fill: '#94a3b8'}} stroke="rgba(255,255,255,0.1)" />
                    
                    <Tooltip contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', border: '1px solid var(--border-color)', borderRadius: '8px', color: '#fff' }} formatter={(value, name, props) => {
                      if (name === "高値・安値" && Array.isArray(value)) return [`安値: ${value[0].toLocaleString()}円 / 高値: ${value[1].toLocaleString()}円`, name];
                      if (name === "始値・終値" && Array.isArray(value)) {
                        const isUp = props.payload.isUp;
                        return [`${isUp ? '陽線 (上昇)' : '陰線 (下落)'} (始値: ${props.payload.open.toLocaleString()}円 / 終値: ${props.payload.close.toLocaleString()}円)`, name];
                      }
                      if (typeof value === 'number') return [`${value.toLocaleString()}円`, name];
                      return [value, name];
                    }} />
                    
                    {chartType === 'line' && <Line type="monotone" dataKey="close" stroke="var(--accent-cyan)" strokeWidth={2} dot={false} name="終値" />}
                    {chartType === 'candle' && <Bar dataKey="lowHigh" barSize={2} fill="#64748b" name="高値・安値" />}
                    {chartType === 'candle' && (
                      <Bar dataKey="openClose" barSize={8} name="始値・終値">
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.isUp ? 'var(--up-color)' : 'var(--down-color)'} />
                        ))}
                      </Bar>
                    )}
                    
                    {showSMA5 && <Line type="monotone" dataKey="sma5" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="5日SMA" />}
                    {showSMA25 && <Line type="monotone" dataKey="sma25" stroke="#84cc16" strokeWidth={1.5} dot={false} name="25日SMA" />}
                    {showBB && <Line type="monotone" dataKey="bbUpper" stroke="rgba(139, 92, 246, 0.5)" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="+2σ" />}
                    {showBB && <Line type="monotone" dataKey="bbLower" stroke="rgba(139, 92, 246, 0.5)" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="-2σ" />}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* サブチャート群 */}
            <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
              {/* RSI Chart */}
              {showRSI && (
                <div style={{ flex: '1', minWidth: '280px', backgroundColor: 'var(--panel-bg)', padding: '10px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                  <h3 style={{ fontSize: '13px', margin: '0 0 10px 0', color: 'var(--text-muted)' }}>RSI (14日)</h3>
                  <div style={{ height: '120px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 0, left: -20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" hide />
                        <YAxis domain={[0, 100]} tick={{fontSize: 10, fill: '#94a3b8'}} stroke="none" />
                        <Tooltip contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', border: 'none', borderRadius: '4px' }} />
                        {/* 買われすぎ/売られすぎライン */}
                        <Line type="step" dataKey={() => 70} stroke="rgba(239, 68, 68, 0.3)" strokeWidth={1} dot={false} isAnimationActive={false} name="買われすぎ(70)" />
                        <Line type="step" dataKey={() => 30} stroke="rgba(6, 182, 212, 0.3)" strokeWidth={1} dot={false} isAnimationActive={false} name="売られすぎ(30)" />
                        <Line type="monotone" dataKey="rsi" stroke="var(--accent-pink)" strokeWidth={1.5} dot={false} name="RSI" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* MACD Chart */}
              {showMACD && (
                <div style={{ flex: '1', minWidth: '280px', backgroundColor: 'var(--panel-bg)', padding: '10px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                  <h3 style={{ fontSize: '13px', margin: '0 0 10px 0', color: 'var(--text-muted)' }}>MACD (12, 26, 9)</h3>
                  <div style={{ height: '120px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData} margin={{ top: 5, right: 10, bottom: 0, left: -20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" hide />
                        <YAxis tick={{fontSize: 10, fill: '#94a3b8'}} stroke="none" />
                        <Tooltip contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', border: 'none', borderRadius: '4px' }} />
                        <Bar dataKey={(d) => d.macd - d.macdSignal} fill="var(--accent-purple)" opacity={0.5} name="ヒストグラム" />
                        <Line type="monotone" dataKey="macd" stroke="var(--accent-cyan)" strokeWidth={1.5} dot={false} name="MACD" />
                        <Line type="monotone" dataKey="macdSignal" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Signal" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>

            {/* 出来高 */}
            <div style={{ backgroundColor: 'var(--panel-bg)', padding: '10px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
              <h3 style={{ fontSize: '13px', margin: '0 0 10px 0', color: 'var(--text-muted)' }}>出来高</h3>
              <div style={{ height: '80px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
                    <XAxis dataKey="date" hide />
                    <Tooltip contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', border: 'none', borderRadius: '4px' }} />
                    <Bar dataKey="volume" fill="rgba(248, 250, 252, 0.2)" name="出来高" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

          </div>

          {/* 右側：企業財務・サマリーパネル (30%) */}
          <div style={{ flex: '1 1 30%', minWidth: '300px' }}>
            <div style={{ backgroundColor: 'var(--panel-bg)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border-color)', backdropFilter: 'blur(10px)', height: '100%' }}>
              <h2 style={{ margin: '0 0 5px 0', fontSize: '22px', color: 'var(--text-main)' }}>
                {stockInfo ? stockInfo.name : '取得中...'}
              </h2>
              <p style={{ margin: '0 0 20px 0', fontSize: '14px', color: 'var(--accent-cyan)' }}>コード: {selectedCode}</p>

              {stockInfo ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                  
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>前日終値</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                      {stockInfo.previousClose ? `¥${stockInfo.previousClose.toLocaleString()}` : '-'}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>時価総額</div>
                      <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                        {stockInfo.marketCap ? `¥${Math.round(stockInfo.marketCap / 100000000).toLocaleString()}億円` : '-'}
                      </div>
                    </div>
                    
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>配当利回り</div>
                      <div style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--accent-pink)' }}>
                        {stockInfo.dividendYield ? `${(stockInfo.dividendYield * 100).toFixed(2)}%` : '-'}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>PER (実績)</div>
                      <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                        {stockInfo.trailingPE ? `${stockInfo.trailingPE.toFixed(1)}倍` : '-'}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>PBR (実績)</div>
                      <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                        {stockInfo.priceToBook ? `${stockInfo.priceToBook.toFixed(2)}倍` : '-'}
                      </div>
                    </div>
                  </div>

                  <div style={{ marginTop: '10px', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>52週レンジ (高値 - 安値)</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '14px', fontWeight: 'bold' }}>
                      <span style={{ color: 'var(--down-color)' }}>¥{stockInfo.fiftyTwoWeekLow?.toLocaleString() || '-'}</span>
                      <span style={{ color: 'var(--text-muted)' }}>—</span>
                      <span style={{ color: 'var(--up-color)' }}>¥{stockInfo.fiftyTwoWeekHigh?.toLocaleString() || '-'}</span>
                    </div>
                  </div>

                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>財務データを読み込んでいます...</div>
              )}
            </div>
          </div>
          
        </div>
      )}
    </div>
  );
}
