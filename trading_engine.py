#!/usr/bin/env python3
"""
LIVE CRYPTO TRADING SIMULATOR - VERSIÓN CORREGIDA Y OPTIMIZADA
Escanea 50 criptomonedas en timeframes 3m/5m/4h, usa módulos externos,
trailing stop avanzado y consola en vivo. Persistencia de estado y log de trades.
"""

import os
import sys
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------
SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "MATIC",
    "DOT", "TRX", "LTC", "BCH", "APT", "ARB", "OP", "NEAR", "ATOM", "FIL",
    "SUI", "RNDR", "INJ", "STX", "GRT", "ALGO", "EGLD", "AAVE", "FTM", "KAS",
    "PEPE", "SHIB", "SEI", "TIA", "PYTH", "JTO", "BLUR", "IMX", "RUNE", "DYDX",
    "COMP", "SNX", "MKR", "UNI", "CRV", "GMX", "LDO", "ENS", "WLD", "BONK"
]
TF_ENTRY = "3m"
TF_CONF = "5m"
TF_TREND = "4h"

CAPITAL_INICIAL = 1000.0
RIESGO_POR_TRADE = 0.01          # 1% del capital
SL_PCT = 0.003                    # 0.3% stop loss fijo
TP_PCT = 0.005                    # 0.5% take profit fijo

# Exchanges en orden de preferencia (los primeros tienen prioridad)
EXCHANGE_NAMES = ['kucoin', 'bybit', 'binance', 'kraken']
EXCHANGES = []
for name in EXCHANGE_NAMES:
    try:
        exchange_class = getattr(ccxt, name)
        exch = exchange_class({
            'enableRateLimit': True,
            'timeout': 15000,
        })
        EXCHANGES.append(exch)
    except (AttributeError, Exception) as e:
        print(f"⚠️ No se pudo inicializar exchange {name}: {e}")

# CoinGecko es opcional para tickers, si no está disponible se omite
try:
    COINGECKO = ccxt.coingecko({
        'enableRateLimit': True,
        'timeout': 15000,
    })
except AttributeError:
    COINGECKO = None
    print("⚠️ CoinGecko no disponible, se usará solo los exchanges principales para precios.")

CACHE_TTL = 30                     # segundos
LOOP_INTERVAL = 60                  # segundos

# -------------------------------------------------------------------
# MÓDULOS EXTERNOS (placeholders – el usuario debe reemplazar con los reales)
# -------------------------------------------------------------------
def REC_MO(df_3m: pd.DataFrame) -> float:
    """Retorna score entre -1 y 1 (positivo = largos, negativo = cortos)"""
    if len(df_3m) < 14:
        return 0.0
    close = df_3m['close'].values
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0).mean()
    loss = -np.where(delta < 0, delta, 0).mean()
    if loss == 0:
        return 1.0
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return (rsi - 50) / 50.0

def BP_Delta_Edge(df_3m: pd.DataFrame) -> float:
    """Retorna fuerza de la señal (0..1)"""
    if len(df_3m) < 20:
        return 0.0
    close = df_3m['close'].values
    ema = pd.Series(close).ewm(span=20).mean().values
    slope = (ema[-1] - ema[-5]) / ema[-5] if len(ema) >= 5 else 0
    return min(abs(slope) * 10, 1.0)

def Scalp_Detector(df_3m: pd.DataFrame) -> bool:
    return True

def Scalp_Validator(df_3m: pd.DataFrame) -> bool:
    return True

# -------------------------------------------------------------------
# TRAILING STOP AVANZADO
# -------------------------------------------------------------------
class Direction(Enum):
    LONG = 1
    SHORT = -1

@dataclass
class Trade:
    id: int
    symbol: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: float
    risk_points: float

    highest_price: float = field(init=False)
    lowest_price: float = field(init=False)
    trailing_active: bool = False
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        if self.direction == Direction.LONG:
            self.take_profit = self.entry_price * (1 + TP_PCT)
        else:
            self.take_profit = self.entry_price * (1 - TP_PCT)

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None

    @property
    def pnl(self) -> float:
        if not self.is_closed:
            return 0.0
        if self.direction == Direction.LONG:
            return (self.exit_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.exit_price) * self.size

    @property
    def R(self) -> Optional[float]:
        if not self.is_closed:
            return None
        if self.direction == Direction.LONG:
            return (self.exit_price - self.entry_price) / self.risk_points
        else:
            return (self.entry_price - self.exit_price) / self.risk_points

    @property
    def risk_amount(self) -> float:
        return self.size * self.risk_points

    def to_dict(self) -> dict:
        d = asdict(self)
        d['direction'] = self.direction.value
        d['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            d['exit_time'] = self.exit_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'Trade':
        data['direction'] = Direction(data['direction'])
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if data.get('exit_time'):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        trade = cls(**data)
        trade.__post_init__()
        return trade

class ValidatedTrailingEngine:
    def __init__(self, config: Dict):
        self.initial_capital = config["initial_capital"]
        self.cash = self.initial_capital
        self.risk_per_trade = config["risk_per_trade"]
        self.max_total_risk = config.get("max_total_risk", 0.05)
        self.trailing_mult = config["trailing_distance_mult"]
        self.trailing_activation_R = config.get("trailing_activation_R", 1.0)

        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.next_id = 0
        self.equity_curve = []
        self._logged_ids = set()

    def _total_open_risk(self) -> float:
        return sum(t.risk_amount for t in self.open_trades)

    def open_trade(self, symbol: str, timestamp: datetime, entry_price: float,
                   direction: Direction) -> Optional[int]:
        if direction == Direction.LONG:
            stop_loss = entry_price * (1 - SL_PCT)
        else:
            stop_loss = entry_price * (1 + SL_PCT)

        risk_points = abs(entry_price - stop_loss)
        if risk_points <= 0:
            return None

        risk_amount = self.cash * self.risk_per_trade
        if self._total_open_risk() + risk_amount > self.cash * self.max_total_risk:
            return None

        size = risk_amount / risk_points

        trade = Trade(
            id=self.next_id,
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            risk_points=risk_points
        )
        self.open_trades.append(trade)
        self.next_id += 1
        return trade.id

    def update_bar(self, symbol: str, timestamp: datetime, high: float, low: float,
                   close: float, atr: float):
        to_close = []
        for trade in self.open_trades:
            if trade.symbol != symbol:
                continue

            # Verificar take profit fijo
            if trade.direction == Direction.LONG:
                if high >= trade.take_profit:
                    self._close_trade(trade, timestamp, trade.take_profit)
                    to_close.append(trade)
                    continue
            else:
                if low <= trade.take_profit:
                    self._close_trade(trade, timestamp, trade.take_profit)
                    to_close.append(trade)
                    continue

            if trade.direction == Direction.LONG:
                trade.highest_price = max(trade.highest_price, high)

                if not trade.trailing_active:
                    if high >= trade.entry_price + self.trailing_activation_R * trade.risk_points:
                        trade.trailing_active = True

                if trade.trailing_active:
                    new_stop = trade.highest_price - self.trailing_mult * atr
                    trade.stop_loss = max(trade.stop_loss, new_stop)

                if low <= trade.stop_loss:
                    self._close_trade(trade, timestamp, trade.stop_loss)
                    to_close.append(trade)

            else:
                trade.lowest_price = min(trade.lowest_price, low)

                if not trade.trailing_active:
                    if low <= trade.entry_price - self.trailing_activation_R * trade.risk_points:
                        trade.trailing_active = True

                if trade.trailing_active:
                    new_stop = trade.lowest_price + self.trailing_mult * atr
                    trade.stop_loss = min(trade.stop_loss, new_stop)

                if high >= trade.stop_loss:
                    self._close_trade(trade, timestamp, trade.stop_loss)
                    to_close.append(trade)

        for t in to_close:
            self.open_trades.remove(t)

        unrealized = 0.0
        for t in self.open_trades:
            if t.direction == Direction.LONG:
                unrealized += (close - t.entry_price) * t.size
            else:
                unrealized += (t.entry_price - close) * t.size
        equity = self.cash + unrealized
        self.equity_curve.append((timestamp, equity))

    def _close_trade(self, trade: Trade, timestamp: datetime, exit_price: float):
        if trade.direction == Direction.LONG:
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size
        self.cash += pnl
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        self.closed_trades.append(trade)

    def to_dict(self) -> dict:
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'risk_per_trade': self.risk_per_trade,
            'max_total_risk': self.max_total_risk,
            'trailing_mult': self.trailing_mult,
            'trailing_activation_R': self.trailing_activation_R,
            'next_id': self.next_id,
            'open_trades': [t.to_dict() for t in self.open_trades],
            'closed_trades': [t.to_dict() for t in self.closed_trades],
            'equity_curve': [(t.isoformat(), v) for t, v in self.equity_curve],
            '_logged_ids': list(self._logged_ids)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ValidatedTrailingEngine':
        engine = cls({
            'initial_capital': data['initial_capital'],
            'risk_per_trade': data['risk_per_trade'],
            'max_total_risk': data['max_total_risk'],
            'trailing_distance_mult': data['trailing_mult'],
            'trailing_activation_R': data['trailing_activation_R']
        })
        engine.cash = data['cash']
        engine.next_id = data['next_id']
        engine.open_trades = [Trade.from_dict(t) for t in data['open_trades']]
        engine.closed_trades = [Trade.from_dict(t) for t in data['closed_trades']]
        engine.equity_curve = [(datetime.fromisoformat(t), v) for t, v in data['equity_curve']]
        engine._logged_ids = set(data.get('_logged_ids', []))
        return engine

# -------------------------------------------------------------------
# GESTIÓN DE DATOS CON CACHÉ Y FALLBACK
# -------------------------------------------------------------------
_ohlcv_cache = {}

def fetch_ohlcv_with_cache(symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    symbol_usdt = f"{symbol}/USDT"
    now = datetime.now()

    for exchange in EXCHANGES:
        cache_key = (exchange.id, symbol, timeframe)
        if cache_key in _ohlcv_cache:
            ts, df = _ohlcv_cache[cache_key]
            if (now - ts).seconds < CACHE_TTL:
                return df

        try:
            if not exchange.markets:
                exchange.load_markets()
            ohlcv = None
            if symbol_usdt in exchange.markets:
                ohlcv = exchange.fetch_ohlcv(symbol_usdt, timeframe, limit=limit)
            else:
                alt = f"{symbol}USDT"
                if alt in exchange.markets:
                    ohlcv = exchange.fetch_ohlcv(alt, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 20:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                _ohlcv_cache[cache_key] = (now, df)
                return df
        except Exception as e:
            continue
    return None

def get_current_price(symbol: str) -> Optional[float]:
    symbol_usdt = f"{symbol}/USDT"
    # Primero intentar con exchanges principales
    for exchange in EXCHANGES:
        try:
            if not exchange.markets:
                exchange.load_markets()
            if symbol_usdt in exchange.markets:
                ticker = exchange.fetch_ticker(symbol_usdt)
                return ticker['last']
            alt = f"{symbol}USDT"
            if alt in exchange.markets:
                ticker = exchange.fetch_ticker(alt)
                return ticker['last']
        except:
            continue
    # Si no, intentar con CoinGecko si está disponible
    if COINGECKO is not None:
        try:
            ticker = COINGECKO.fetch_ticker(f"{symbol}/usd")
            return ticker['last']
        except:
            pass
    return None

# -------------------------------------------------------------------
# INDICADORES
# -------------------------------------------------------------------
def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_atr(df: pd.DataFrame, period=14) -> float:
    if len(df) < period + 1:
        return np.nan
    high, low, close = df['high'], df['low'], df['close']
    tr = np.maximum(high - low,
                    np.abs(high - close.shift()),
                    np.abs(low - close.shift()))
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr

def volume_sma(df: pd.DataFrame, period=20) -> float:
    return df['volume'].rolling(period).mean().iloc[-1]

# -------------------------------------------------------------------
# ANÁLISIS DE SEÑAL
# -------------------------------------------------------------------
def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    df_4h = fetch_ohlcv_with_cache(symbol, TF_TREND, 100)
    if df_4h is None or len(df_4h) < 50:
        return None
    macd_4h, signal_4h, _ = compute_macd(df_4h)
    trend_long = macd_4h.iloc[-1] > signal_4h.iloc[-1]
    trend_short = macd_4h.iloc[-1] < signal_4h.iloc[-1]

    df_5m = fetch_ohlcv_with_cache(symbol, TF_CONF, 60)
    if df_5m is None or len(df_5m) < 30:
        return None
    _, _, hist_5m = compute_macd(df_5m)
    hist_prev = hist_5m.iloc[-2]
    hist_now = hist_5m.iloc[-1]
    cross_up = hist_now > 0 and hist_prev <= 0
    cross_down = hist_now < 0 and hist_prev >= 0

    df_3m = fetch_ohlcv_with_cache(symbol, TF_ENTRY, 100)
    if df_3m is None or len(df_3m) < 30:
        return None

    vol_sma = volume_sma(df_3m)
    current_vol = df_3m['volume'].iloc[-1]
    volume_ok = current_vol > vol_sma

    recmo_score = REC_MO(df_3m)
    bp_edge = BP_Delta_Edge(df_3m)
    scalp_detected = Scalp_Detector(df_3m)
    scalp_valid = Scalp_Validator(df_3m)

    possible_long = trend_long and cross_up and scalp_detected and scalp_valid and volume_ok
    possible_short = trend_short and cross_down and scalp_detected and scalp_valid and volume_ok

    if not (possible_long or possible_short):
        return None

    vol_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0
    hist_strength = abs(hist_now) / df_5m['close'].iloc[-1] * 100
    hist_strength_norm = min(hist_strength / 2.0, 1.0)

    if possible_long:
        score = (bp_edge * 0.3 +
                 abs(recmo_score) * 0.3 +
                 min(vol_ratio, 3) / 3.0 * 0.2 +
                 hist_strength_norm * 0.2)
        direction = Direction.LONG
    else:
        score = (bp_edge * 0.3 +
                 abs(recmo_score) * 0.3 +
                 min(vol_ratio, 3) / 3.0 * 0.2 +
                 hist_strength_norm * 0.2)
        direction = Direction.SHORT

    if direction == Direction.SHORT and recmo_score > 0:
        score *= 0.5
    if direction == Direction.LONG and recmo_score < 0:
        score *= 0.5

    return {
        'symbol': symbol,
        'direction': direction,
        'score': score,
        'entry_price': df_3m['close'].iloc[-1],
        'atr': compute_atr(df_3m, 14),
        'timestamp': df_3m['timestamp'].iloc[-1]
    }

# -------------------------------------------------------------------
# PERSISTENCIA Y LOG
# -------------------------------------------------------------------
STATE_FILE = 'state.json'
TRADES_LOG = 'trades_log.txt'

def save_state(engine: ValidatedTrailingEngine):
    with open(STATE_FILE, 'w') as f:
        json.dump(engine.to_dict(), f, indent=2, default=str)

def load_state() -> Optional[ValidatedTrailingEngine]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        return ValidatedTrailingEngine.from_dict(data)
    return None

def log_trade(trade: Trade, engine: ValidatedTrailingEngine):
    if trade.id in engine._logged_ids:
        return
    with open(TRADES_LOG, 'a') as f:
        f.write(f"{trade.exit_time.isoformat()},{trade.symbol},{trade.direction.name},"
                f"{trade.entry_price:.4f},{trade.exit_price:.4f},{trade.stop_loss:.4f},"
                f"{trade.pnl:.2f},{trade.R:.2f},{engine.cash:.2f}\n")
    engine._logged_ids.add(trade.id)

# -------------------------------------------------------------------
# CONSOLA EN VIVO
# -------------------------------------------------------------------
def display_console(engine: ValidatedTrailingEngine, best_signal: Optional[Dict] = None):
    in_github = os.getenv('GITHUB_ACTIONS') == 'true'
    if not in_github:
        print("\033[H\033[J", end="")

    print("=" * 52)
    print("          LIVE TRADING SIMULATOR")
    print("=" * 52)
    print(f"\nCapital: {engine.cash:.2f} USDT")
    print(f"Trade activo: {len(engine.open_trades)}")

    if engine.open_trades:
        t = engine.open_trades[0]
        price = get_current_price(t.symbol) or t.entry_price
        if t.direction == Direction.LONG:
            pnl_pct = (price - t.entry_price) / t.entry_price * 100
        else:
            pnl_pct = (t.entry_price - price) / t.entry_price * 100

        print(f"\nSYMBOL: {t.symbol}USDT")
        print(f"SIDE: {t.direction.name}")
        print("")
        print(f"ENTRY: {t.entry_price:.2f}")
        print(f"PRICE: {price:.2f}")
        print("")
        print(f"TP: {t.take_profit:.2f}")
        print(f"SL: {t.stop_loss:.2f}")
        print("")
        print(f"TRAILING STOP: {t.stop_loss:.2f}")
        print(f"TRAIL ACTIVE: {str(t.trailing_active).upper()}")
        print("")
        print(f"PNL: {pnl_pct:+.2f}%")
    else:
        print("\nNo hay trade activo")
        if best_signal:
            print(f"Mejor señal: {best_signal['symbol']} {best_signal['direction'].name} "
                  f"Score: {best_signal['score']:.2f}")

    print("=" * 52)
    print(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------------------------------------------------------
# LOOP PRINCIPAL
# -------------------------------------------------------------------
def main_loop():
    engine = load_state()
    if engine is None:
        engine = ValidatedTrailingEngine({
            'initial_capital': CAPITAL_INICIAL,
            'risk_per_trade': RIESGO_POR_TRADE,
            'max_total_risk': 0.05,
            'trailing_distance_mult': 1.5,
            'trailing_activation_R': 1.0
        })

    for exchange in EXCHANGES:
        try:
            exchange.load_markets()
        except Exception as e:
            print(f"⚠️ Error cargando mercado de {exchange.id}: {e}")

    while True:
        loop_start = time.time()
        try:
            best_signal = None
            best_score = -1

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_sym = {executor.submit(analyze_symbol, sym): sym for sym in SYMBOLS}
                for future in as_completed(future_to_sym):
                    try:
                        result = future.result(timeout=30)
                        if result and result['score'] > best_score:
                            best_score = result['score']
                            best_signal = result
                    except Exception as e:
                        sym = future_to_sym[future]
                        print(f"Error analizando {sym}: {e}")

            if len(engine.open_trades) == 0 and best_signal:
                trade_id = engine.open_trade(
                    symbol=best_signal['symbol'],
                    timestamp=best_signal['timestamp'],
                    entry_price=best_signal['entry_price'],
                    direction=best_signal['direction']
                )
                if trade_id is not None:
                    print(f"✅ Trade abierto: {best_signal['symbol']} {best_signal['direction'].name}")

            if engine.open_trades:
                symbols_with_trades = set(t.symbol for t in engine.open_trades)
                for sym in symbols_with_trades:
                    df = fetch_ohlcv_with_cache(sym, TF_ENTRY, 50)
                    if df is not None and len(df) > 20:
                        atr = compute_atr(df)
                        last = df.iloc[-1]
                        engine.update_bar(
                            symbol=sym,
                            timestamp=last['timestamp'],
                            high=last['high'],
                            low=last['low'],
                            close=last['close'],
                            atr=atr
                        )

            for trade in engine.closed_trades:
                if trade.id not in engine._logged_ids:
                    log_trade(trade, engine)

            display_console(engine, best_signal)
            save_state(engine)

            elapsed = time.time() - loop_start
            sleep_time = max(1, LOOP_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Detenido por usuario")
            save_state(engine)
            break
        except Exception as e:
            print(f"❌ Error en loop principal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
