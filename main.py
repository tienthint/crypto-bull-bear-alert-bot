

import requests
import pandas as pd
import ta
from datetime import datetime
import pytz

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = '7510157645:AAEuMk6ymG1JWIW1wQXCGqkLb_xdjXxEFnA'
CHAT_ID = '6849082725'

# === TRACK LAST SIGNALS IN MEMORY (OPTIONAL: Persist if needed) ===
last_signals = {}



# === TIME HELPER ===
def format_singapore_time():
    sg_timezone = pytz.timezone("Asia/Singapore")
    return datetime.now(sg_timezone).strftime('%Y-%m-%d %H:%M')

# === TELEGRAM SEND ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        print(f"📨 Telegram response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Telegram send error: {e}")

# === TELEGRAM TEST ===
def test_telegram():
    send_telegram_message("✅ Test message: Bot is connected and running.")

def get_top_binance_symbols(limit=10):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    try:
        tickers = response.json()
    except Exception as e:
        print("❌ Failed to parse Binance response:", e)
        return []
    if not isinstance(tickers, list):
        print("❌ Unexpected Binance response:", tickers)
        return []

    spot_pairs = [t for t in tickers if t['symbol'].endswith("USDT") and float(t['quoteVolume']) > 1_000_000]
    sorted_pairs = sorted(spot_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_pairs[:limit]]

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

def check_signals(symbol, df):
    try:
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=100).mean()
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
        return f"🚀 [BULL RUN DETECTED] {symbol} @ {format_singapore_time()} | Price: ${latest['close']:.4f}\n" + "\n".join(messages)

    messages = []

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
        return f"📉 [BEAR RUN DETECTED] {symbol} @ {format_singapore_time()} | Price: ${latest['close']:.4f}\n" + "\n".join(messages)

    return None

# === MAIN JOB ===
def job():
    global last_signals
    print(f"Running check at {format_singapore_time()}")
    symbols = get_top_binance_symbols()

    for symbol in symbols:
        try:
            df = fetch_price_data(symbol)
            signal = check_signals(symbol, df)

            if signal:
                print(f"✅ Signal detected for {symbol}")
                send_telegram_message(f"{symbol}: {signal}")
            else:
                print(f"ℹ️ No signal for {symbol} at {format_singapore_time()}")

        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")

# === ENTRY POINT ===
if __name__ == "__main__":
    print("Bot started... 🚀")
    test_telegram()  # Send test message on start
    job()  # Run signal detection once
