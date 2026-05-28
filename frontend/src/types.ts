/**
 * 共通TypeScript型定義
 *
 * APIレスポンスとコンポーネント間で共有される型を定義。
 * `any` 型の使用を最小限に抑え、型安全性を向上させる。
 */

/** 株価データ（APIレスポンスの1行） */
export interface StockData {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number;
  sma5: number | null;
  sma25: number | null;
  bbUpper: number | null;
  bbLower: number | null;
  rsi: number | null;
  macd: number | null;
  macdSignal: number | null;
}

/** チャート表示用に整形された株価データ */
export interface ChartDataItem extends StockData {
  lowHigh: [number | null, number | null];
  openClose: [number, number];
  isUp: boolean;
  bbLowerUpper: [number, number] | null;
  type: 'historical' | 'prediction';
  prediction: number | null;
}

/** 企業情報 */
export interface StockInfo {
  code: string;
  name: string;
  marketCap: number | null;
  trailingPE: number | null;
  priceToBook: number | null;
  dividendYield: number | null;
  fiftyTwoWeekHigh: number | null;
  fiftyTwoWeekLow: number | null;
  previousClose: number | null;
}

/** AI予測レスポンス */
export interface PredictionResponse {
  code: string;
  prediction: number | null;
  predictions: number[];
  mape: number | null;
  message: string;
}

/** 学習ステータス */
export interface TrainingStatus {
  status: 'idle' | 'training' | 'success' | 'failed';
  progress: number;
  message: string;
  prediction?: number;
  predictions?: number[];
  mape?: number | null;
}

/** ウォッチリストアイテム */
export interface WatchlistItem {
  code: string;
  name: string;
}
