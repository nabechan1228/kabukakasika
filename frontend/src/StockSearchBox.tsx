import { useState, useEffect, useRef, useMemo } from 'react';
// 用意したマスターデータをインポート
import stockMaster from './stock_master.json';

interface StockSearchBoxProps {
  onSelect: (code: string) => void;
  initialCode?: string;
}

export default function StockSearchBox({ onSelect, initialCode }: StockSearchBoxProps) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [selectedStock, setSelectedStock] = useState<{ code: string; name: string } | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // 初期値のセット
  useEffect(() => {
    if (initialCode) {
      const stock = stockMaster.find(s => s.code === initialCode);
      if (stock) {
        setSelectedStock(stock);
        setQuery(`${stock.name} (${stock.code})`);
      }
    }
  }, [initialCode]);

  // ▼ 高速フィルタリング処理 (useMemoで入力のたびに再計算)
  const filteredStocks = useMemo(() => {
    const selectedText = selectedStock ? `${selectedStock.name} (${selectedStock.code})` : '';
    // 未入力、または現在選択されている銘柄の表示テキストと完全に一致している場合は全件表示（最初の50件）
    if (!query || query === selectedText) {
      return stockMaster.slice(0, 50);
    }
    
    const lowerQuery = query.toLowerCase();
    return stockMaster.filter(stock => 
      stock.name.toLowerCase().includes(lowerQuery) || 
      stock.code.includes(lowerQuery)
    ).slice(0, 50); // DOMの過負荷を防ぐため最大50件に制限
  }, [query, selectedStock]);

  // ▼ 画面外クリックでドロップダウンを閉じる処理
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (code: string, name: string) => {
    setSelectedStock({ code, name });
    setQuery(`${name} (${code})`); // 選択した銘柄を入力欄に反映
    setIsOpen(false);
    onSelect(code); // 親コンポーネントに選択されたコードを渡す
  };

  return (
    <div ref={wrapperRef} style={{ position: 'relative', width: '300px' }}>
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setIsOpen(true);
        }}
        onClick={(e) => {
          (e.target as HTMLInputElement).select(); // クリック時にも全選択する
        }}
        placeholder="銘柄名またはコード (例: ソニー, 6758)"
        style={{
          width: '100%',
          padding: '10px 15px',
          fontSize: '16px',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
          backgroundColor: 'rgba(15, 23, 42, 0.6)',
          color: 'var(--text-main)',
          boxSizing: 'border-box',
          outline: 'none',
          boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
          transition: 'all 0.3s ease'
        }}
        onFocus={(e) => {
          setIsOpen(true);
          e.target.select(); // フォーカス時にテキストを全選択する（上書き入力しやすくするため）
          e.target.style.border = '1px solid var(--accent-cyan)';
          e.target.style.boxShadow = '0 0 10px rgba(6, 182, 212, 0.5)';
        }}
        onBlur={(e) => {
          e.target.style.border = '1px solid var(--border-color)';
          e.target.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
        }}
      />
      
      {/* ドロップダウンリスト */}
      {isOpen && filteredStocks.length > 0 && (
        <ul style={{
          position: 'absolute',
          top: 'calc(100% + 5px)',
          left: 0,
          right: 0,
          margin: 0,
          padding: 0,
          listStyle: 'none',
          backgroundColor: 'var(--bg-dark)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          maxHeight: '300px',
          overflowY: 'auto',
          zIndex: 1000,
          boxShadow: '0 10px 15px -3px rgba(0,0,0,0.5)',
          backdropFilter: 'blur(10px)'
        }}>
          {filteredStocks.map((stock) => (
            <li
              key={stock.code}
              onClick={() => handleSelect(stock.code, stock.name)}
              style={{
                padding: '10px 15px',
                cursor: 'pointer',
                borderBottom: '1px solid var(--border-color)',
                color: 'var(--text-main)',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
            >
              <strong>{stock.name}</strong> <span style={{ color: 'var(--text-muted)', fontSize: '0.9em' }}>({stock.code})</span>
            </li>
          ))}
        </ul>
      )}
      {isOpen && query && query !== (selectedStock ? `${selectedStock.name} (${selectedStock.code})` : '') && filteredStocks.length === 0 && (
        <div style={{ position: 'absolute', top: 'calc(100% + 5px)', left: 0, right: 0, padding: '10px', background: 'var(--bg-dark)', border: '1px solid var(--border-color)', zIndex: 1000, borderRadius: '8px', color: 'var(--text-muted)' }}>
          見つかりませんでした
        </div>
      )}
    </div>
  );
}
