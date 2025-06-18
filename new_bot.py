import pandas as pd
import numpy as np
import requests
import ta
import schedule
import time
from datetime import datetime
from threading import Thread
import logging

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = '7510157645:AAEuMk6ymG1JWIW1wQXCGqkLb_xdjXxEFnA'
CHAT_ID = '6849082725'

last_signals = {}

def send_telegram_message(message):
    """Send message to Telegram with error handling"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        logger.info(f"Message sent to Telegram: {message[:50]}...")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

def get_top_symbols_by_interest(limit=10):
    """Get top trading symbols by volume"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        tickers = response.json()
        
        # Filter for liquid USDT pairs with sufficient volume
        filtered = [
            t for t in tickers
            if t['symbol'].endswith("USDT")
            and float(t['quoteVolume']) > 1_000_000
            and not any(x in t['symbol'] for x in ['BUSD', 'FDUSD'])
        ]
        
        # Sort by volume and get top symbols
        filtered.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        top_symbols = [t['symbol'] for t in filtered[:limit]]
        
        logger.info(f"Top {limit} symbols: {top_symbols}")
        return top_symbols
        
    except Exception as e:
        logger.error(f"Error fetching top symbols: {e}")
        # Return default symbols if API fails
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']

def fetch_price_data(symbol="BTCUSDT", interval="30m", limit=200):
    """Fetch price data from Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", 
            "close_time", "quote_volume", "count", "taker_buy_volume", 
            "taker_buy_quote_volume", "ignore"
        ])
        
        # Convert to appropriate data types
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["open"] = df["open"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {e}")
        return None

def get_funding_rate(symbol='BTCUSDT'):
    """Get current funding rate for futures"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['fundingRate'])
        return 0.0
    except Exception as e:
        logger.warning(f"Error fetching funding rate for {symbol}: {e}")
        return 0.0

def calculate_fib_levels(df, window=30):
    """Calculate Fibonacci retracement levels"""
    try:
        recent = df.tail(window)
        max_p = recent['high'].max()
        min_p = recent['low'].min()
        diff = max_p - min_p
        
        return {
            '0.236': max_p - 0.236 * diff,
            '0.382': max_p - 0.382 * diff,
            '0.5': max_p - 0.5 * diff,
            '0.618': max_p - 0.618 * diff,
            '0.786': max_p - 0.786 * diff
        }
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return {}

def get_multi_timeframe_data(symbol):
    """Get data from multiple timeframes for better trend analysis"""
    timeframes = {
        '15m': fetch_price_data(symbol, '15m', 100),
        '1h': fetch_price_data(symbol, '1h', 100),
        '4h': fetch_price_data(symbol, '4h', 100)
    }
    return timeframes

def analyze_trend_strength(df):
    """Analyze overall trend strength and direction"""
    try:
        # Calculate multiple EMAs for trend analysis
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        current = df.iloc[-1]
        prev_10 = df.iloc[-10] if len(df) >= 10 else df.iloc[0]
        
        # Price vs EMAs (trend direction)
        ema_alignment = 0
        if pd.notna(current['ema_20']) and pd.notna(current['ema_50']):
            if current['close'] > current['ema_20'] > current['ema_50']:
                ema_alignment = 2  # Strong uptrend
            elif current['close'] > current['ema_20']:
                ema_alignment = 1  # Mild uptrend
            elif current['close'] < current['ema_20'] < current['ema_50']:
                ema_alignment = -2  # Strong downtrend
            elif current['close'] < current['ema_20']:
                ema_alignment = -1  # Mild downtrend
        
        # Price momentum (% change over period)
        price_momentum = 0
        if pd.notna(current['close']) and pd.notna(prev_10['close']):
            price_change = (current['close'] - prev_10['close']) / prev_10['close'] * 100
            if price_change > 5:
                price_momentum = 2
            elif price_change > 2:
                price_momentum = 1
            elif price_change < -5:
                price_momentum = -2
            elif price_change < -2:
                price_momentum = -1
        
        return ema_alignment, price_momentum, price_change
        
    except Exception as e:
        logger.error(f"Error analyzing trend strength: {e}")
        return 0, 0, 0

def detect_bottom_top_signals(symbol, timeframe_data, funding_rate, fib):
    """Detect bottom (BUY) and top (SELL) signals with enhanced analysis"""
    try:
        df_15m = timeframe_data['15m']
        df_1h = timeframe_data['1h'] 
        df_4h = timeframe_data['4h']
        
        if any(df is None or len(df) < 50 for df in [df_15m, df_1h, df_4h]):
            return None, None
        
        # Calculate comprehensive indicators
        for df in [df_15m, df_1h, df_4h]:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.BollingerBands(
                df['close'], window=20).bollinger_hband(), ta.volatility.BollingerBands(
                df['close'], window=20).bollinger_mavg(), ta.volatility.BollingerBands(
                df['close'], window=20).bollinger_lband()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        
        # Stochastic for 15m
        stoch = ta.momentum.StochasticOscillator(
            high=df_15m['high'], low=df_15m['low'], close=df_15m['close'], window=14
        )
        df_15m['stoch_k'] = stoch.stoch()
        df_15m['stoch_d'] = stoch.stoch_signal()
        
        current_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        prev2_15m = df_15m.iloc[-3]
        
        current_1h = df_1h.iloc[-1]
        prev_1h = df_1h.iloc[-2]
        
        current_4h = df_4h.iloc[-1]
        prev_4h = df_4h.iloc[-2]
        
        price = current_15m['close']
        
        # =============================================================================
        # BOTTOM DETECTION (BUY SIGNALS)
        # =============================================================================
        
        bottom_conditions = []
        bottom_score = 0
        
        # 1. MULTI-TIMEFRAME OVERSOLD REVERSAL (Strong Signal)
        oversold_15m = pd.notna(current_15m['rsi']) and current_15m['rsi'] < 30
        oversold_1h = pd.notna(current_1h['rsi']) and current_1h['rsi'] < 35
        oversold_4h = pd.notna(current_4h['rsi']) and current_4h['rsi'] < 40
        
        rsi_reversal_15m = (pd.notna(current_15m['rsi']) and pd.notna(prev_15m['rsi']) and 
                           current_15m['rsi'] > prev_15m['rsi'] + 2)
        
        if oversold_15m and oversold_1h and rsi_reversal_15m:
            bottom_conditions.append(f"🔥 Multi-TF oversold reversal (RSI: {current_15m['rsi']:.1f}/15m, {current_1h['rsi']:.1f}/1h)")
            bottom_score += 4
        elif oversold_15m and rsi_reversal_15m:
            bottom_conditions.append(f"Oversold bounce (RSI: {current_15m['rsi']:.1f})")
            bottom_score += 2
        
        # 2. STOCHASTIC OVERSOLD CROSS (Medium Signal)
        if (pd.notna(current_15m['stoch_k']) and pd.notna(current_15m['stoch_d']) and
            pd.notna(prev_15m['stoch_k']) and pd.notna(prev_15m['stoch_d'])):
            
            stoch_oversold = current_15m['stoch_k'] < 25
            stoch_cross_up = (current_15m['stoch_k'] > current_15m['stoch_d'] and 
                             prev_15m['stoch_k'] <= prev_15m['stoch_d'])
            
            if stoch_oversold and stoch_cross_up:
                bottom_conditions.append(f"Stochastic bullish cross from oversold ({current_15m['stoch_k']:.1f})")
                bottom_score += 2
        
        # 3. BOLLINGER BAND SQUEEZE + BREAKOUT (Strong Signal)
        if (pd.notna(current_15m['bb_lower']) and pd.notna(prev_15m['bb_lower'])):
            bb_touch = price <= current_15m['bb_lower'] * 1.02  # Within 2% of lower band
            bb_bounce = price > prev_15m['close']  # Price bouncing up
            
            if bb_touch and bb_bounce:
                bottom_conditions.append("Bollinger Band lower touch + bounce")
                bottom_score += 3
        
        # 4. VOLUME SPIKE ON DECLINE (Capitulation Signal)
        if (pd.notna(current_15m['volume']) and pd.notna(current_15m['volume_sma'])):
            volume_spike = current_15m['volume'] > current_15m['volume_sma'] * 1.5
            price_decline = current_15m['close'] < prev_15m['close']
            
            if volume_spike and price_decline and rsi_reversal_15m:
                bottom_conditions.append("Volume capitulation + reversal")
                bottom_score += 3
        
        # 5. MACD DIVERGENCE (Medium Signal)
        if (pd.notna(current_15m['macd']) and pd.notna(prev_15m['macd']) and 
            pd.notna(current_1h['macd']) and pd.notna(prev_1h['macd'])):
            
            macd_cross_15m = (current_15m['macd'] > current_15m['macd_signal'] and 
                             prev_15m['macd'] <= prev_15m['macd_signal'])
            macd_positive_1h = current_1h['macd'] > prev_1h['macd']
            
            if macd_cross_15m and macd_positive_1h:
                bottom_conditions.append("MACD bullish cross + momentum")
                bottom_score += 2
        
        # 6. FIBONACCI SUPPORT + FUNDING RATE (Strong Signal)
        if fib and '0.786' in fib and '0.618' in fib:
            deep_support = price <= fib['0.786'] * 1.02
            funding_support = funding_rate < -0.01
            
            if deep_support and funding_support:
                bottom_conditions.append(f"Deep Fib support + negative funding ({funding_rate:.4f})")
                bottom_score += 3
            elif deep_support:
                bottom_conditions.append("Deep Fibonacci support level")
                bottom_score += 2
        
        # 7. TREND REVERSAL PATTERN (Medium Signal)
        if (pd.notna(current_15m['ema_20']) and pd.notna(current_15m['ema_50'])):
            price_above_ema20 = price > current_15m['ema_20']
            ema20_above_ema50 = current_15m['ema_20'] > current_15m['ema_50']
            prev_below = prev_15m['close'] < prev_15m['ema_20']
            
            if price_above_ema20 and prev_below:
                if ema20_above_ema50:
                    bottom_conditions.append("Strong trend reversal (price back above EMA20)")
                    bottom_score += 2
                else:
                    bottom_conditions.append("Potential trend reversal (price above EMA20)")
                    bottom_score += 1
        
        # =============================================================================
        # TOP DETECTION (SELL SIGNALS)
        # =============================================================================
        
        top_conditions = []
        top_score = 0
        
        # 1. MULTI-TIMEFRAME OVERBOUGHT REVERSAL (Strong Signal)
        overbought_15m = pd.notna(current_15m['rsi']) and current_15m['rsi'] > 70
        overbought_1h = pd.notna(current_1h['rsi']) and current_1h['rsi'] > 65
        overbought_4h = pd.notna(current_4h['rsi']) and current_4h['rsi'] > 60
        
        rsi_reversal_15m = (pd.notna(current_15m['rsi']) and pd.notna(prev_15m['rsi']) and 
                           current_15m['rsi'] < prev_15m['rsi'] - 2)
        
        if overbought_15m and overbought_1h and rsi_reversal_15m:
            top_conditions.append(f"🔥 Multi-TF overbought reversal (RSI: {current_15m['rsi']:.1f}/15m, {current_1h['rsi']:.1f}/1h)")
            top_score += 4
        elif overbought_15m and rsi_reversal_15m:
            top_conditions.append(f"Overbought rejection (RSI: {current_15m['rsi']:.1f})")
            top_score += 2
        
        # 2. STOCHASTIC OVERBOUGHT CROSS (Medium Signal)
        if (pd.notna(current_15m['stoch_k']) and pd.notna(current_15m['stoch_d']) and
            pd.notna(prev_15m['stoch_k']) and pd.notna(prev_15m['stoch_d'])):
            
            stoch_overbought = current_15m['stoch_k'] > 75
            stoch_cross_down = (current_15m['stoch_k'] < current_15m['stoch_d'] and 
                               prev_15m['stoch_k'] >= prev_15m['stoch_d'])
            
            if stoch_overbought and stoch_cross_down:
                top_conditions.append(f"Stochastic bearish cross from overbought ({current_15m['stoch_k']:.1f})")
                top_score += 2
        
        # 3. BOLLINGER BAND UPPER REJECTION (Strong Signal)
        if (pd.notna(current_15m['bb_upper']) and pd.notna(prev_15m['bb_upper'])):
            bb_touch = price >= current_15m['bb_upper'] * 0.98  # Within 2% of upper band
            bb_rejection = price < prev_15m['close']  # Price rejecting down
            
            if bb_touch and bb_rejection:
                top_conditions.append("Bollinger Band upper rejection")
                top_score += 3
        
        # 4. VOLUME SPIKE ON RALLY (Exhaustion Signal)
        if (pd.notna(current_15m['volume']) and pd.notna(current_15m['volume_sma'])):
            volume_spike = current_15m['volume'] > current_15m['volume_sma'] * 1.5
            price_rally = current_15m['close'] > prev_15m['close']
            
            if volume_spike and price_rally and rsi_reversal_15m:
                top_conditions.append("Volume exhaustion + reversal")
                top_score += 3
        
        # 5. MACD BEARISH DIVERGENCE (Medium Signal)
        if (pd.notna(current_15m['macd']) and pd.notna(prev_15m['macd']) and 
            pd.notna(current_1h['macd']) and pd.notna(prev_1h['macd'])):
            
            macd_cross_15m = (current_15m['macd'] < current_15m['macd_signal'] and 
                             prev_15m['macd'] >= prev_15m['macd_signal'])
            macd_negative_1h = current_1h['macd'] < prev_1h['macd']
            
            if macd_cross_15m and macd_negative_1h:
                top_conditions.append("MACD bearish cross + momentum")
                top_score += 2
        
        # 6. FIBONACCI RESISTANCE + FUNDING RATE (Strong Signal)
        if fib and '0.236' in fib and '0.382' in fib:
            strong_resistance = price >= fib['0.236'] * 0.98
            funding_resistance = funding_rate > 0.015
            
            if strong_resistance and funding_resistance:
                top_conditions.append(f"Fib resistance + high funding ({funding_rate:.4f})")
                top_score += 3
            elif strong_resistance:
                top_conditions.append("Strong Fibonacci resistance level")
                top_score += 2
        
        # 7. TREND EXHAUSTION PATTERN (Medium Signal)
        if (pd.notna(current_15m['ema_20']) and pd.notna(current_15m['ema_50'])):
            price_below_ema20 = price < current_15m['ema_20']
            prev_above = prev_15m['close'] > prev_15m['ema_20']
            
            if price_below_ema20 and prev_above:
                top_conditions.append("Trend exhaustion (price below EMA20)")
                top_score += 2
        
        # =============================================================================
        # SIGNAL GENERATION
        # =============================================================================
        
        signal = None
        signal_type = None
        
        # BOTTOM (BUY) SIGNAL
        if bottom_score >= 5 and len(bottom_conditions) >= 2:
            strength = "🔥 STRONG" if bottom_score >= 8 else "⚡ MODERATE"
            confidence = min(95, 60 + (bottom_score * 4))
            
            signal = (f"🟢 *{strength} BOTTOM* - {symbol} 🏆\n"
                     f"💰 BUY ZONE: ${price:.4f}\n"
                     f"⭐ Confidence: {confidence}% | Score: {bottom_score}/12\n"
                     f"🎯 Action: CONSIDER BUYING\n"
                     f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"📊 Bottom Signals:\n• " + "\n• ".join(bottom_conditions))
            signal_type = "bottom"
            
        # TOP (SELL) SIGNAL  
        elif top_score >= 5 and len(top_conditions) >= 2:
            strength = "🔥 STRONG" if top_score >= 8 else "⚡ MODERATE"
            confidence = min(95, 60 + (top_score * 4))
            
            signal = (f"🔴 *{strength} TOP* - {symbol} 🏆\n"
                     f"💰 SELL ZONE: ${price:.4f}\n"
                     f"⭐ Confidence: {confidence}% | Score: {top_score}/12\n"
                     f"🎯 Action: CONSIDER SELLING\n"
                     f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"📊 Top Signals:\n• " + "\n• ".join(top_conditions))
            signal_type = "top"

        return signal, signal_type
        
    except Exception as e:
        logger.error(f"Error detecting bottom/top for {symbol}: {e}")
        return None, None

def analyze_symbol(symbol):
    """Analyze a single symbol for bottom/top signals with multi-timeframe data"""
    try:
        logger.info(f"Analyzing {symbol}")
        
        # Fetch multi-timeframe data
        timeframe_data = get_multi_timeframe_data(symbol)
        
        # Check if we have sufficient data
        if any(df is None or len(df) < 50 for df in timeframe_data.values()):
            logger.warning(f"Insufficient data for {symbol}")
            return None, None
            
        funding = get_funding_rate(symbol)
        fibs = calculate_fib_levels(timeframe_data['15m'])  # Use 15m for Fibonacci
        
        # Detect bottom/top signals with enhanced analysis
        signal, signal_type = detect_bottom_top_signals(symbol, timeframe_data, funding, fibs)
        
        return signal, signal_type
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None, None

def job():
    """Main job function that runs periodically"""
    global last_signals
    
    logger.info(f"🔍 Running analysis at {datetime.now()}")
    
    try:
        symbols = get_top_symbols_by_interest(limit=8)  # Get top 8 coins by volume
        signals_found = []
        
        # Send summary of coins being analyzed (first run or every hour)
        current_hour = datetime.now().hour
        if not hasattr(job, 'last_summary_hour') or job.last_summary_hour != current_hour:
            summary_msg = (f"📊 *TOP COINS ANALYSIS*\n"
                          f"🕐 {datetime.now().strftime('%H:%M:%S')}\n"
                          f"📈 Analyzing top {len(symbols)} coins by volume:\n"
                          f"💎 {' • '.join(symbols[:5])}\n"
                          f"🔍 Scanning for bull/bear signals...")
            send_telegram_message(summary_msg)
            job.last_summary_hour = current_hour
        
        for symbol in symbols:
            try:
                signal, signal_type = analyze_symbol(symbol)
                
                # Check if signal has changed
                previous_signal = last_signals.get(symbol)
                
                if signal and signal_type != previous_signal:
                    logger.info(f"New {signal_type} signal for {symbol}")
                    
                    send_telegram_message(signal)
                    
                    # Format signal type for summary
                    if signal_type == "bottom":
                        signals_found.append(f"{symbol} (🟢 BUY)")
                    elif signal_type == "top":
                        signals_found.append(f"{symbol} (🔴 SELL)")
                    
                    last_signals[symbol] = signal_type
                elif signal_type is None and previous_signal is not None:
                    # Signal cleared
                    last_signals[symbol] = None
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Send summary if signals were found
        if signals_found:
            summary = (f"🎯 *TRADING SIGNALS SUMMARY*\n"
                      f"📊 Found {len(signals_found)} new signals:\n"
                      f"• {chr(10).join(signals_found)}\n\n"
                      f"💡 *Quick Guide:*\n"
                      f"🟢 BOTTOM = Consider Buying\n"
                      f"🔴 TOP = Consider Selling")
            send_telegram_message(summary)
                
        logger.info(f"Analysis complete - {len(signals_found)} signals found")
        
    except Exception as e:
        logger.error(f"Error in main job: {e}")

def test_connection():
    """Test Telegram connection"""
    try:
        test_message = (f"🤖 *TRADING BOT STARTED*\n"
                       f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                       f"🎯 Monitoring TOP coins by volume\n"
                       f"📊 Analysis every 15 minutes\n"
                       f"✅ System operational")
        send_telegram_message(test_message)
        return True
    except Exception as e:
        logger.error(f"Telegram connection test failed: {e}")
        return False

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 Enhanced Trading Bot Starting...")
    
    # Test connections
    logger.info("Testing Telegram connection...")
    if test_connection():
        logger.info("✅ Telegram connection successful")
    else:
        logger.warning("⚠️ Telegram connection failed")
    
    # Test API connection
    logger.info("Testing Binance API connection...")
    test_symbols = get_top_symbols_by_interest(limit=3)
    if test_symbols:
        logger.info("✅ Binance API connection successful")
    else:
        logger.error("❌ Binance API connection failed")
        exit(1)
    
    # Schedule job every 15 minutes
    schedule.every(15).minutes.do(job)
    
    # Run first analysis
    logger.info("Running initial analysis...")
    job()
    
    # Main loop
    logger.info("✅ Bot is now running. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        send_telegram_message("🤖 Trading Bot Stopped\n👋 See you later!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        send_telegram_message(f"🚨 Bot Error: {str(e)}")