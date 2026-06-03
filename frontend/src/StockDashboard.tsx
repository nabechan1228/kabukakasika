import { useState, useEffect, useMemo, useRef } from 'react';
import type { StockData, StockInfo, WatchlistItem, TrainingStatus } from './types';
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
  Cell,
  Area,
  ReferenceArea
} from 'recharts';

import StockSearchBox from './StockSearchBox';

// APIベースURL (環境変数から取得、デフォルトはローカル開発用)
// Windows環境でlocalhostがIPv6 (::1) に解決されて接続エラー(Failed to fetch)になるのを防ぐため、127.0.0.1を使用します
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// =========================================================
// カスタムツールチップ (金融ターミナル風のプロフェッショナルな情報表示)
// =========================================================
const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: Record<string, any> }> }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isPred = data.type === 'prediction';
    
    return (
      <div style={{
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        border: '1px solid var(--border-color)',
        padding: '12px 16px',
        borderRadius: '8px',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.7)',
        backdropFilter: 'blur(8px)',
        fontSize: '12px',
        color: '#f8fafc',
        minWidth: '220px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '6px', marginBottom: '6px', fontWeight: 'bold' }}>
          <span style={{ color: 'var(--text-main)' }}>📅 {data.date}</span>
          {isPred ? (
            <span style={{ color: '#a855f7', background: 'rgba(168, 85, 247, 0.2)', padding: '2px 6px', borderRadius: '4px', fontSize: '10px' }}>AI予測</span>
          ) : (
            <span style={{ color: 'var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.1)', padding: '2px 6px', borderRadius: '4px', fontSize: '10px' }}>実績</span>
          )}
        </div>
        
        {/* 株価４本値のグリッド */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 12px', marginBottom: '8px' }}>
          <div>始値: <strong style={{ float: 'right' }}>{data.open ? `¥${Math.round(data.open).toLocaleString()}` : '-'}</strong></div>
          <div>終値: <strong style={{ float: 'right', color: isPred ? '#a855f7' : (data.isUp ? 'var(--up-color)' : 'var(--down-color)') }}>{data.close ? `¥${Math.round(data.close).toLocaleString()}` : '-'}</strong></div>
          <div>高値: <strong style={{ float: 'right' }}>{data.high ? `¥${Math.round(data.high).toLocaleString()}` : '-'}</strong></div>
          <div>安値: <strong style={{ float: 'right' }}>{data.low ? `¥${Math.round(data.low).toLocaleString()}` : '-'}</strong></div>
        </div>

        <div style={{ borderTop: '1px dashed rgba(255,255,255,0.1)', paddingTop: '6px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {!isPred ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>出来高:</span>
                <strong>{data.volume ? `${data.volume.toLocaleString()} 株` : '0 株'}</strong>
              </div>
              {data.sma5 && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#f59e0b' }}>● 5日SMA:</span>
                  <strong>¥{Math.round(data.sma5).toLocaleString()}</strong>
                </div>
              )}
              {data.sma25 && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#84cc16' }}>● 25日SMA:</span>
                  <strong>¥{Math.round(data.sma25).toLocaleString()}</strong>
                </div>
              )}
              {data.bbUpper && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'rgba(139, 92, 246, 0.7)' }}>● ボリンジャー[+2σ]:</span>
                  <strong>¥{Math.round(data.bbUpper).toLocaleString()}</strong>
                </div>
              )}
              {data.bbLower && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'rgba(139, 92, 246, 0.7)' }}>● ボリンジャー[-2σ]:</span>
                  <strong>¥{Math.round(data.bbLower).toLocaleString()}</strong>
                </div>
              )}
              {data.rsi && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--accent-pink)' }}>● RSI(14):</span>
                  <strong style={{ color: data.rsi >= 70 ? 'var(--up-color)' : (data.rsi <= 30 ? 'var(--accent-cyan)' : '#fff') }}>{data.rsi.toFixed(1)}%</strong>
                </div>
              )}
              {data.macd && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--accent-cyan)' }}>● MACD:</span>
                  <strong>{data.macd.toFixed(1)}</strong>
                </div>
              )}
            </>
          ) : (
            <div style={{ color: '#a855f7', textAlign: 'center', marginTop: '4px', fontWeight: 'bold' }}>
              🤖 AI予測終値: ¥{Math.round(data.close).toLocaleString()}
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

export default function StockDashboard() {
  const [selectedCode, setSelectedCode] = useState<string>("7203");
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [mape, setMape] = useState<number | null>(null);
  const [training, setTraining] = useState<boolean>(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({ status: 'idle', progress: 0, message: '' });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // コンポーネントunmount時にポーリングをクリーンアップ
  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);
  
  const [chartType, setChartType] = useState<'line' | 'candle'>('candle');

  // 指標トグル
  const [showSMA5, setShowSMA5] = useState<boolean>(true);
  const [showSMA25, setShowSMA25] = useState<boolean>(true);
  const [showBB, setShowBB] = useState<boolean>(true);
  const [showRSI, setShowRSI] = useState<boolean>(true);   // RSI
  const [showMACD, setShowMACD] = useState<boolean>(true); // MACD

  // ウォッチリストのロード（データ破損に対する安全なパース）
  useEffect(() => {
    const saved = localStorage.getItem('kabukakasika_watchlist');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          setWatchlist(parsed);
        }
      } catch {
        // データが破損している場合は無視して初期値を使用
        localStorage.removeItem('kabukakasika_watchlist');
      }
    }
  }, []);

  const handleToggleWatchlist = () => {
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

  // ▼ 実績データと予測データを結合し、ボリンジャーバンド用レンジも計算する
  const chartData = useMemo(() => {
    if (stockData.length === 0) return [];

    // 1. 既存の過去実績データを整形
    const formattedData = stockData.map(item => ({
      ...item,
      lowHigh: [item.low, item.high],
      openClose: [Math.min(item.open, item.close), Math.max(item.open, item.close)],
      isUp: item.close >= item.open,
      bbLowerUpper: item.bbLower && item.bbUpper ? [item.bbLower, item.bbUpper] : null,
      type: 'historical',
      prediction: null // 過去データは予測線表示用には基本null
    }));

    // 2. 予測データがある場合、実績データの末尾から連結する
    if (predictions && predictions.length > 0) {
      const lastItem = formattedData[formattedData.length - 1];
      
      // 折れ線が実績の終値からシームレスに繋がるよう、実績の最終日の予測値（prediction）に実績終値をセット
      formattedData[formattedData.length - 1].prediction = lastItem.close;

      const lastDate = new Date(lastItem.date);

      const predictionItems = predictions.map((predPrice, idx) => {
        // 土日を除く未来の営業日を計算
        let nextDate = new Date(lastDate);
        let daysCount = 0;
        while (daysCount < idx + 1) {
          nextDate.setDate(nextDate.getDate() + 1);
          const day = nextDate.getDay();
          if (day !== 0 && day !== 6) {
            daysCount++;
          }
        }

        const dateStr = nextDate.toISOString().split('T')[0];

        return {
          date: `${dateStr} (予)`,
          close: predPrice,
          open: idx === 0 ? lastItem.close : predictions[idx - 1], // 直前の終値を今日の始値とする
          low: predPrice,
          high: predPrice,
          volume: 0,
          type: 'prediction',
          prediction: predPrice,
          isUp: idx === 0 ? predPrice >= lastItem.close : predPrice >= predictions[idx - 1],
          bbLowerUpper: null
        };
      });

      return [...formattedData, ...predictionItems];
    }

    return formattedData;
  }, [stockData, predictions]);

  useEffect(() => {
    const fetchStockData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_BASE}/api/stock/${selectedCode}`);
        if (!response.ok) throw new Error('データの取得に失敗しました');
        const data = await response.json();
        setStockData(data);

        // 企業情報の取得
        try {
          const infoRes = await fetch(`${API_BASE}/api/info/${selectedCode}`);
          if (infoRes.ok) setStockInfo(await infoRes.json());
          else setStockInfo(null);
        } catch (e) {
          setStockInfo(null);
        }

        // AI予測の取得
        try {
          const predResponse = await fetch(`${API_BASE}/api/predict/${selectedCode}`);
          if (predResponse.ok) {
            const predData = await predResponse.json();
            setPrediction(predData.prediction);
            setPredictions(predData.predictions || []);
            setMape(predData.mape || null);
          } else {
            setPrediction(null);
            setPredictions([]);
          }
        } catch (e) {
          setPrediction(null);
          setPredictions([]);
          setMape(null);
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchStockData();
  }, [selectedCode]);

  // 非同期学習ポーリングロジックの導入
  const handleTrainModel = async () => {
    setTraining(true);
    setError(null);
    setTrainingStatus({ status: 'training', progress: 0, message: 'AI学習タスクを開始中...' });

    try {
      // 前回のポーリングが残っていればクリア
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }

      const response = await fetch(`${API_BASE}/api/train/${selectedCode}`, { method: 'POST' });
      if (!response.ok) throw new Error('AIの学習開始に失敗しました');
      
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/api/train/status/${selectedCode}`);
          if (!statusRes.ok) return;
          const statusData = await statusRes.json();
          
          setTrainingStatus({
            status: statusData.status,
            progress: statusData.progress,
            message: statusData.message
          });

          if (statusData.status === 'success') {
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
            setPredictions(statusData.predictions || []);
            setPrediction(statusData.prediction);
            setMape(statusData.mape || null);
            setTraining(false);
          } else if (statusData.status === 'failed') {
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
            setError(statusData.message || 'AI学習プロセスでエラーが発生しました');
            setTraining(false);
          }
        } catch (pollErr) {
          console.error("Status polling failed:", pollErr);
        }
      }, 800);
      
    } catch (err: any) {
      setError(err.message);
      setTraining(false);
    }
  };

  // AI予測エリアの表示用X軸範囲の取得
  const predStartDate = useMemo(() => {
    const predItem = chartData.find(d => d.type === 'prediction');
    return predItem ? predItem.date : undefined;
  }, [chartData]);

  const predEndDate = useMemo(() => {
    if (chartData.length === 0) return undefined;
    const lastItem = chartData[chartData.length - 1];
    return lastItem.type === 'prediction' ? lastItem.date : undefined;
  }, [chartData]);

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
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', flexWrap: 'wrap', gap: '10px' }}>
                <h2 style={{ fontSize: '16px', margin: 0, color: 'var(--text-muted)' }}>株価推移・テクニカル指標</h2>
                
                {training ? (
                  /* ⚡ AI学習中 プログレスバーUI */
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', width: '220px', background: 'rgba(15,23,42,0.8)', padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 'bold' }}>
                      <span style={{ color: 'var(--accent-cyan)' }}>⚡ AI学習中...</span>
                      <span>{trainingStatus.progress}%</span>
                    </div>
                    <div style={{ width: '100%', height: '5px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${trainingStatus.progress}%`, height: '100%', background: 'linear-gradient(90deg, #06b6d4, #8b5cf6)', borderRadius: '3px', transition: 'width 0.2s ease' }} />
                    </div>
                    <span style={{ fontSize: '9px', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{trainingStatus.message}</span>
                  </div>
                ) : (prediction !== null && !isNaN(prediction)) ? (
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <div style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)', padding: '6px 14px', borderRadius: '20px', color: '#fff', fontWeight: 'bold', fontSize: '13px', boxShadow: '0 0 10px rgba(236, 72, 153, 0.4)' }}>
                      🤖 AI予測 (翌日終値): {Math.round(prediction).toLocaleString()} 円
                    </div>
                    {mape !== null && !isNaN(mape) && (
                      <div style={{
                        padding: '4px 10px', borderRadius: '12px', fontSize: '11px', fontWeight: 'bold',
                        background: mape < 2 ? 'rgba(16,185,129,0.2)' : mape < 5 ? 'rgba(245,158,11,0.2)' : 'rgba(239,68,68,0.2)',
                        color: mape < 2 ? '#10b981' : mape < 5 ? '#f59e0b' : '#ef4444',
                        border: `1px solid ${mape < 2 ? 'rgba(16,185,129,0.4)' : mape < 5 ? 'rgba(245,158,11,0.4)' : 'rgba(239,68,68,0.4)'}`
                      }}>
                        精度: MAPE {mape.toFixed(1)}%
                      </div>
                    )}
                    <button 
                      onClick={handleTrainModel} 
                      style={{ 
                        background: 'rgba(255,255,255,0.05)', 
                        border: '1px solid var(--border-color)', 
                        padding: '6px 12px', 
                        borderRadius: '20px', 
                        color: 'var(--text-main)', 
                        fontSize: '12px', 
                        cursor: 'pointer',
                        transition: 'background-color 0.2s'
                      }} 
                      title="モデルを再学習します"
                    >
                      再学習
                    </button>
                  </div>
                ) : (
                  <button onClick={handleTrainModel} style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', border: 'none', padding: '6px 14px', borderRadius: '20px', color: '#fff', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px', boxShadow: '0 0 10px rgba(16, 185, 129, 0.4)' }}>
                    🤖 AI予測: 未学習 [学習開始]
                  </button>
                )}
              </div>

              <div style={{ height: '400px', width: '100%', position: 'relative' }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <ComposedChart data={chartData} barGap="-100%" margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" tick={{fontSize: 11, fill: '#94a3b8'}} stroke="rgba(255,255,255,0.1)" />
                    <YAxis domain={['auto', 'auto']} tick={{fontSize: 11, fill: '#94a3b8'}} stroke="rgba(255,255,255,0.1)" />
                    
                    <Tooltip content={<CustomTooltip />} />

                    {/* AI予測期間エリアの半透明グラデーション背景 */}
                    {predStartDate && predEndDate && (
                      <ReferenceArea 
                        x1={predStartDate} 
                        x2={predEndDate} 
                        fill="rgba(168, 85, 247, 0.05)" 
                        stroke="rgba(168, 85, 247, 0.2)"
                        strokeDasharray="3 3"
                      />
                    )}

                    {/* ボリンジャーバンド半透明エリア塗りつぶし */}
                    {showBB && (
                      <Area 
                        type="monotone" 
                        dataKey="bbLowerUpper" 
                        fill="rgba(139, 92, 246, 0.03)" 
                        stroke="none" 
                        name="ボリンジャーバンド"
                        legendType="none"
                      />
                    )}
                    
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
                    
                    {/* ボリンジャーバンド上限・下限ライン */}
                    {showBB && <Line type="monotone" dataKey="bbUpper" stroke="rgba(139, 92, 246, 0.4)" strokeWidth={1.2} strokeDasharray="4 4" dot={false} name="+2σ" />}
                    {showBB && <Line type="monotone" dataKey="bbLower" stroke="rgba(139, 92, 246, 0.4)" strokeWidth={1.2} strokeDasharray="4 4" dot={false} name="-2σ" />}

                    {/* AI予測折れ線 (ネオンパープルの破線) */}
                    <Line 
                      type="monotone" 
                      dataKey="prediction" 
                      stroke="#a855f7" 
                      strokeWidth={2} 
                      strokeDasharray="4 4" 
                      dot={{ r: 3.5, fill: '#a855f7', strokeWidth: 0 }}
                      name="AI予測終値" 
                      connectNulls
                    />
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
                  <div style={{ height: '120px', width: '100%', position: 'relative' }}>
                    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
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
                  <div style={{ height: '120px', width: '100%', position: 'relative' }}>
                    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
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
              <div style={{ height: '80px', width: '100%', position: 'relative' }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <BarChart data={chartData} margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
                    <XAxis dataKey="date" hide />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="volume" name="出来高">
                      {chartData.map((entry, index) => (
                        <Cell 
                          key={`cell-vol-${index}`} 
                          fill={entry.close >= entry.open ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'} 
                        />
                      ))}
                    </Bar>
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
