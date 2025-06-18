import os
import requests
import pandas as pd
import ta
import schedule
import time
from datetime import datetime
import pytz

# === TIMEZONE CONFIG ===
SINGAPORE_TZ = pytz.timezone('Asia/Singapore')

def get_singapore_time():
    """Get current time in Singapore timezone"""
    return datetime.now(SINGAPORE_TZ)

def format_singapore_time():
    """Get formatted Singapore time string"""
    return get_singapore_time().strftime('%Y-%m-%d %H:%M:%S SGT')
# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = '7510157645:AAEuMk6ymG1JWIW1wQXCGqkLb_xdjXxEFnA'
CHAT_ID = '6849082725'


# === TRACK LAST SIGNALS ===
last_signals = {}

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("ERROR: Telegram credentials not set!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print(f"Failed to send message: {response.status_code}")
    except Exception as e:
        print(f"Error sending message: {e}")

# === FETCH TOP 10 SYMBOLS BY VOLUME ===
def get_top_binance_symbols(limit=10):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    tickers = requests.get(url).json()

    # Filter for liquid USDT pairs with sufficient volume
    filtered = [
        t for t in tickers
        if t['symbol'].endswith("USDT")
        and float(t['quoteVolume']) > 1_000_000
        and not any(x in t['symbol'] for x in ['BUSD', 'FDUSD'])  # optional filters
    ]

    # Score each symbol by interest: price change % x log(volume)
    for t in filtered:
        try:
            t['score'] = abs(float(t['priceChangePercent'])) * (float(t['quoteVolume']) ** 0.2)
        except:
            t['score'] = 0

    # Sort by score descending
    sorted_by_interest = sorted(filtered, key=lambda x: t['score'], reverse=True)
    
    return [t['symbol'] for t in sorted_by_interest[:limit]]

# === FETCH PRICE DATA ===
def fetch_price_data(symbol="BTCUSDT", interval="15m", limit=200):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"
        ])
        df["open"] = df["open"].astype(float)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# === SIGNAL DETECTION ===
def check_signals(symbol, df):
    try:
        if df is None or len(df) < 200:
            return None
            
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
        previous = df.iloc[-2]

    except Exception as e:
        print(f"[{symbol}] Error in signal processing: {e}")
        return None

    messages = []
    bull_count = 0
    bear_count = 0

    # Bull rules
    if latest['close'] > latest['open']:
        bull_count += 1
        messages.append("✔️ Green Candle (Bullish)")
    messages.append("❌ Red Candle – Cancels Bull signal")
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
        return (
        f"🚀 [BULL RUN DETECTED] {symbol} @ {format_singapore_time()}\n"
        f"💰 Price: ${latest['close']:.4f}\n"
        + "\n".join(messages)
    )

    # Reset messages
    messages = []

    # Bear rules
    if latest['close'] < latest['open']:
        bear_count += 1
        messages.append("❌ Red Candle (Bearish)")
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
        return (
        f"📉 [BEAR RUN DETECTED] {symbol} @ {format_singapore_time()}\n"
        f"💰 Price: ${latest['close']:.4f}\n"
        + "\n".join(messages)
    )

    return None

# === JOB LOOP ===
def job():
    global last_signals
    print(f"Running check at {format_singapore_time()}")
    symbols = get_top_binance_symbols()
    print(f"Checking symbols: {symbols}")

    for symbol in symbols:
        try:
            df = fetch_price_data(symbol)
            if df is None:
                continue
                
            signal = check_signals(symbol, df)

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

            # Send message only if signal changes
            if last_signals.get(symbol) != current_signal:
                last_signals[symbol] = current_signal
                if signal:
                    print(f"New signal for {symbol}: {current_signal}")
                    send_telegram_message(signal)
                    
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    print(f"Check completed at {format_singapore_time()}")

# === MAIN EXECUTION ===
def main():
    print("🚀 Crypto Bot Started!")
    print(f"Bot started at: {format_singapore_time()}")
    print(f"Bot will check every 15 minutes...")
    
    # Test telegram connection
    if TELEGRAM_TOKEN and CHAT_ID:
        send_telegram_message(f"🤖 Crypto Bot is now online and monitoring the markets!\n⏰ Started at: {format_singapore_time()}")
    else:
        print("⚠️  WARNING: Telegram credentials not found!")
    
    # Run first check immediately
    job()
    
    # Schedule regular checks
    schedule.every(15).minutes.do(job)
    
    # Keep the bot running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled jobs
        except KeyboardInterrupt:
            print("Bot stopped by user")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(60)  # Wait before continuing

if __name__ == "__main__":
    main()