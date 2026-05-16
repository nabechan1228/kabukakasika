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
  const wrapperRef = useRef<HTMLDivElement>(null);

  // 初期値のセット
  useEffect(() => {
    if (initialCode) {
      const stock = stockMaster.find(s => s.code === initialCode);
      if (stock) setQuery(`${stock.name} (${stock.code})`);
    }
  }, [initialCode]);

  // ▼ 高速フィルタリング処理 (useMemoで入力のたびに再計算)
  const filteredStocks = useMemo(() => {
    if (!query) return stockMaster.slice(0, 50); // 未入力時は最初の50件
    
    const lowerQuery = query.toLowerCase();
    return stockMaster.filter(stock => 
      stock.name.toLowerCase().includes(lowerQuery) || 
      stock.code.includes(lowerQuery)
    ).slice(0, 50); // DOMの過負荷を防ぐため最大50件に制限
  }, [query]);

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
        onFocus={() => setIsOpen(true)}
        placeholder="銘柄名またはコード (例: ソニー, 6758)"
        style={{
          width: '100%',
          padding: '10px',
          fontSize: '16px',
          borderRadius: '4px',
          border: '2px solid red', // デバッグ用に赤枠を追加
          boxSizing: 'border-box'
        }}
      />
      
      {/* ドロップダウンリスト */}
      {isOpen && filteredStocks.length > 0 && (
        <ul style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          margin: 0,
          padding: 0,
          listStyle: 'none',
          backgroundColor: 'white',
          border: '1px solid #ccc',
          borderTop: 'none',
          borderRadius: '0 0 4px 4px',
          maxHeight: '300px',
          overflowY: 'auto',
          zIndex: 1000,
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          {filteredStocks.map((stock) => (
            <li
              key={stock.code}
              onClick={() => handleSelect(stock.code, stock.name)}
              style={{
                padding: '10px',
                cursor: 'pointer',
                borderBottom: '1px solid #eee'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0f8ff'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
            >
              <strong>{stock.name}</strong> <span style={{ color: '#666', fontSize: '0.9em' }}>({stock.code})</span>
            </li>
          ))}
        </ul>
      )}
      {isOpen && query && filteredStocks.length === 0 && (
        <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, padding: '10px', background: 'white', border: '1px solid #ccc', zIndex: 1000 }}>
          見つかりませんでした
        </div>
      )}
    </div>
  );
}
