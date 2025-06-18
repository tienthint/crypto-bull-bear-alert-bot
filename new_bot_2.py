import asyncio
import aiohttp
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoPredictionBot:
    def __init__(self, telegram_bot_token: str, telegram_chat_id: str):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.base_url = "https://api.binance.com/api/v3"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_top_volume_coins(self, limit: int = 10) -> List[str]:
        """Get top coins by 24h volume"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            async with self.session.get(url) as response:
                data = await response.json()
                
            # Filter USDT pairs and sort by volume
            usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
            
            return [pair['symbol'] for pair in sorted_pairs[:limit]]
        except Exception as e:
            logger.error(f"Error fetching top volume coins: {e}")
            return []
    
    async def get_kline_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch kline/candlestick data from Binance"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            logger.error(f"Error fetching kline data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various technical indicators using ta library"""
        if df.empty or len(df) < 50:
            return df
            
        try:
            # Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 
                                                   window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
            
            # Average True Range
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            # Volume indicators
            df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Additional useful indicators from ta library
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def analyze_signals(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Analyze technical signals and generate prediction"""
        if df.empty or len(df) < 50:
            return {'symbol': symbol, 'signal': 'NEUTRAL', 'confidence': 0, 'reasons': []}
        
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            bullish_signals = []
            bearish_signals = []
            confidence_score = 0
            
            # RSI Analysis
            if latest['rsi'] < 30:
                bullish_signals.append("RSI oversold (<30)")
                confidence_score += 15
            elif latest['rsi'] > 70:
                bearish_signals.append("RSI overbought (>70)")
                confidence_score += 15
            
            # MACD Analysis
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                bullish_signals.append("MACD bullish crossover")
                confidence_score += 20
            elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                bearish_signals.append("MACD bearish crossover")
                confidence_score += 20
            
            # Moving Average Analysis
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                bullish_signals.append("Price above MA20 > MA50")
                confidence_score += 10
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                bearish_signals.append("Price below MA20 < MA50")
                confidence_score += 10
            
            # Bollinger Bands Analysis
            if latest['close'] < latest['bb_lower']:
                bullish_signals.append("Price below lower Bollinger Band")
                confidence_score += 15
            elif latest['close'] > latest['bb_upper']:
                bearish_signals.append("Price above upper Bollinger Band")
                confidence_score += 15
            
            # Stochastic Analysis
            if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                bullish_signals.append("Stochastic oversold")
                confidence_score += 10
            elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                bearish_signals.append("Stochastic overbought")
                confidence_score += 10
            
            # Williams %R Analysis (ta library returns values in different range)
            if latest['willr'] < -80:
                bullish_signals.append("Williams %R oversold")
                confidence_score += 8
            elif latest['willr'] > -20:
                bearish_signals.append("Williams %R overbought")
                confidence_score += 8
            
            # ADX for trend strength
            if not pd.isna(latest['adx']) and latest['adx'] > 25:
                confidence_score += 5  # Strong trend adds confidence
            
            # CCI Analysis
            if not pd.isna(latest['cci']):
                if latest['cci'] < -100:
                    bullish_signals.append("CCI oversold")
                    confidence_score += 8
                elif latest['cci'] > 100:
                    bearish_signals.append("CCI overbought")
                    confidence_score += 8
            
            # Volume Analysis
            avg_volume = df['volume'].tail(20).mean()
            if latest['volume'] > avg_volume * 1.5:
                confidence_score += 5
                if bullish_signals:
                    bullish_signals.append("High volume confirmation")
                elif bearish_signals:
                    bearish_signals.append("High volume confirmation")
            
            # Determine final signal
            if len(bullish_signals) > len(bearish_signals) and confidence_score >= 70:
                signal = "BUY"
                reasons = bullish_signals
            elif len(bearish_signals) > len(bullish_signals) and confidence_score >= 70:
                signal = "SELL"
                reasons = bearish_signals
            else:
                signal = "NEUTRAL"
                reasons = bullish_signals + bearish_signals
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': min(confidence_score, 100),
                'reasons': reasons,
                'price': latest['close'],
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'volume': latest['volume']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing signals for {symbol}: {e}")
            return {'symbol': symbol, 'signal': 'NEUTRAL', 'confidence': 0, 'reasons': []}
    
    async def send_telegram_message(self, message: str):
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    logger.info("Telegram message sent successfully")
                else:
                    logger.error(f"Failed to send Telegram message: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def format_signal_message(self, analysis: Dict) -> str:
        """Format analysis result into Telegram message"""
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}
        
        message = f"""
{signal_emoji.get(analysis['signal'], '🟡')} <b>{analysis['symbol']}</b>
        
<b>Signal:</b> {analysis['signal']}
<b>Confidence:</b> {analysis['confidence']}%
<b>Price:</b> ${analysis['price']:.4f}
<b>RSI:</b> {analysis['rsi']:.2f}
<b>MACD:</b> {analysis['macd']:.6f}

<b>Reasons:</b>
{chr(10).join(f"• {reason}" for reason in analysis['reasons'])}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        return message.strip()
    
    async def analyze_all_coins(self) -> List[Dict]:
        """Analyze all top volume coins"""
        top_coins = await self.get_top_volume_coins()
        logger.info(f"Analyzing {len(top_coins)} coins: {top_coins}")
        
        analyses = []
        for symbol in top_coins:
            try:
                df = await self.get_kline_data(symbol)
                if not df.empty:
                    df = self.calculate_technical_indicators(df)
                    analysis = self.analyze_signals(df, symbol)
                    analyses.append(analysis)
                    
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return analyses
    
    async def run_analysis_cycle(self):
        """Run a single analysis cycle"""
        logger.info("Starting analysis cycle...")
        analyses = await self.analyze_all_coins()
        
        # Filter for confident signals
        confident_signals = [
            analysis for analysis in analyses 
            if analysis['signal'] in ['BUY', 'SELL'] and analysis['confidence'] >= 50
        ]
        
        if confident_signals:
            logger.info(f"Found {len(confident_signals)} confident signals")
            
            # Sort by confidence
            confident_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Send notifications for top signals
            for analysis in confident_signals[:5]:  # Top 5 signals
                message = self.format_signal_message(analysis)
                await self.send_telegram_message(message)
                await asyncio.sleep(1)  # Rate limiting for Telegram
        else:
            logger.info("No confident signals found")
    
    async def run_continuously(self, interval_minutes: int = 60):
        """Run the bot continuously"""
        logger.info(f"Starting continuous monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                await self.run_analysis_cycle()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

# Usage example
async def main():
    # Configuration
    TELEGRAM_BOT_TOKEN = "7510157645:AAEuMk6ymG1JWIW1wQXCGqkLb_xdjXxEFnA"
    TELEGRAM_CHAT_ID = "6849082725"
    
    async with CryptoPredictionBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) as bot:
        # Run once
        await bot.run_analysis_cycle()
        
        # Or run continuously (uncomment the line below)
        # await bot.run_continuously(interval_minutes=60)

if __name__ == "__main__":
    asyncio.run(main())