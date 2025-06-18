
from threading import Thread
import requests
import pandas as pd
import ta
import schedule
import time
from datetime import datetime



# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = '7510157645:AAEuMk6ymG1JWIW1wQXCGqkLb_xdjXxEFnA'
CHAT_ID = '6849082725'

# === TRACK LAST SIGNALS ===
last_signals = {}

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

# === FETCH TOP 10 SYMBOLS BY VOLUME ===
def get_top_binance_symbols(limit=10):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    tickers = requests.get(url).json()
    spot_pairs = [t for t in tickers if t['symbol'].endswith("USDT") and float(t['quoteVolume']) > 1_000_000]
    sorted_pairs = sorted(spot_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_pairs[:limit]]

# === FETCH PRICE DATA ===
def fetch_price_data(symbol="BTCUSDT", interval="15m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# === SIGNAL DETECTION ===
def check_signals(symbol, df):
    try:
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd_diff'] = macd.macd_diff()
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
        df['atr'] = atr.average_true_range()
        latest = df.iloc[-1]
    except Exception as e:
        print(f"[{symbol}] Error in signal processing: {e}")
        return None

    messages = []
    bull_count = 0
    bear_count = 0

    # Bull rules
    if latest['close'] > latest['ma200']:
        bull_count += 1
        messages.append("✔️ Price > 200 MA")
    if latest['close'] > latest['ma50']:
        bull_count += 1
        messages.append("✔️ Price > 50 MA")
    if latest['rsi'] > 55:
        bull_count += 1
        messages.append("✔️ RSI > 55")
    if latest['macd'] > latest['macd_signal'] and latest['macd_diff'] > 0:
        bull_count += 1
        messages.append("✔️ MACD Bullish")
    if latest['volume'] > latest['vol_avg']:
        bull_count += 1
        messages.append("✔️ Volume Spike")
    if latest['atr'] > df['atr'].iloc[-2]:
        bull_count += 1
        messages.append("✔️ Rising Volatility")

    if bull_count >= 4:
        return f"🚀 [BULL RUN DETECTED] {symbol} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(messages)

    # Reset messages
    messages = []

    # Bear rules
    if latest['close'] < latest['ma200']:
        bear_count += 1
        messages.append("❌ Price < 200 MA")
    if latest['close'] < latest['ma50']:
        bear_count += 1
        messages.append("❌ Price < 50 MA")
    if latest['rsi'] < 45:
        bear_count += 1
        messages.append("❌ RSI < 45")
    if latest['macd'] < latest['macd_signal'] and latest['macd_diff'] < 0:
        bear_count += 1
        messages.append("❌ MACD Bearish")
    if latest['volume'] > latest['vol_avg']:
        bear_count += 1
        messages.append("❌ Volume Dump")
    if latest['atr'] > df['atr'].iloc[-2]:
        bear_count += 1
        messages.append("❌ Rising Volatility")

    if bear_count >= 4:
        return f"📉 [BEAR RUN DETECTED] {symbol} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(messages)

    return None

# === JOB LOOP ===
def job():
    global last_signals
    print(f"Running check at {datetime.now()}")
    symbols = get_top_binance_symbols()

    for symbol in symbols:
        try:
            df = fetch_price_data(symbol)
            df.name = symbol
            signal = check_signals(symbol,df)

            # Convert signal into a short code for memory (bull/bear/none)
            if signal:
                if "BULL" in signal:
                    current_signal = "bull"
                elif "BEAR" in signal:
                    current_signal = "bear"
                else:
                    current_signal = "unknown"
            else:
                current_signal = None

            # Combine type + message for tracking
            combined_signal = (current_signal, signal)

            # Send message only if signal changes
            if last_signals.get(symbol) != current_signal:
                last_signals[symbol] = current_signal
                if signal:
                    send_telegram_message(f"{symbol}: {signal}")
        except Exception as e:
            print(f"Error for {symbol}: {e}")


schedule.every(15).minutes.do(job)

if __name__ == "__main__":
  print("Bot started... 🚀")
  job()  # ← Add this line to run once right now
  while True:
      schedule.run_pending()
      time.sleep(1)
