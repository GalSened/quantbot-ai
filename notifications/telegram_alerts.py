"""
Telegram notification system for trading alerts.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from ..config.settings import settings


class TelegramNotifier:
    """
    Telegram bot for sending trading notifications and alerts.
    """
    
    def __init__(self):
        self.bot_token = settings.notifications.telegram_bot_token
        self.chat_id = settings.notifications.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("TelegramNotifier initialized successfully")
    
    async def send_message(
        self, 
        message: str, 
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: Message text
            parse_mode: Message formatting (HTML, Markdown)
            disable_notification: Send silently
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Telegram not configured, skipping message")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_notification': disable_notification
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_trade_alert(
        self, 
        symbol: str, 
        action: str, 
        quantity: float, 
        price: float, 
        strategy: str = "Neural AI"
    ) -> bool:
        """
        Send a trade execution alert.
        
        Args:
            symbol: Stock symbol
            action: BUY/SELL
            quantity: Number of shares
            price: Execution price
            strategy: Trading strategy name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            emoji = "🚀" if action.upper() == "BUY" else "💰"
            
            message = f"""
{emoji} <b>NEURAL TRADE EXECUTED</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action.upper()}
<b>Quantity:</b> {quantity:,.0f} shares
<b>Price:</b> ${price:.2f}
<b>Value:</b> ${quantity * price:,.2f}
<b>Strategy:</b> {strategy}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧠 <i>Powered by M3 Max Neural Engine</i>
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
            return False
    
    async def send_portfolio_update(
        self, 
        portfolio_value: float, 
        daily_pnl: float, 
        daily_return: float
    ) -> bool:
        """
        Send portfolio performance update.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily P&L
            daily_return: Daily return percentage
        
        Returns:
            True if successful, False otherwise
        """
        try:
            emoji = "📈" if daily_pnl >= 0 else "📉"
            sign = "+" if daily_pnl >= 0 else ""
            
            message = f"""
{emoji} <b>PORTFOLIO UPDATE</b> {emoji}

<b>Portfolio Value:</b> ${portfolio_value:,.2f}
<b>Daily P&L:</b> {sign}${daily_pnl:,.2f}
<b>Daily Return:</b> {sign}{daily_return:.2f}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧠 <i>Neural AI Trading System</i>
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending portfolio update: {e}")
            return False
    
    async def send_risk_alert(
        self, 
        alert_type: str, 
        symbol: str, 
        current_value: float, 
        threshold: float, 
        message_details: str = ""
    ) -> bool:
        """
        Send risk management alert.
        
        Args:
            alert_type: Type of risk alert
            symbol: Affected symbol
            current_value: Current value
            threshold: Risk threshold
            message_details: Additional details
        
        Returns:
            True if successful, False otherwise
        """
        try:
            message = f"""
🚨 <b>RISK ALERT</b> 🚨

<b>Type:</b> {alert_type}
<b>Symbol:</b> {symbol}
<b>Current:</b> {current_value:.2f}
<b>Threshold:</b> {threshold:.2f}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message_details}

🛡️ <i>Neural Risk Shield Active</i>
            """.strip()
            
            return await self.send_message(message, disable_notification=False)
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
            return False
    
    async def send_market_insight(
        self, 
        insight_type: str, 
        title: str, 
        description: str, 
        confidence: float = 0.0
    ) -> bool:
        """
        Send market insight or prediction.
        
        Args:
            insight_type: Type of insight
            title: Insight title
            description: Detailed description
            confidence: AI confidence level
        
        Returns:
            True if successful, False otherwise
        """
        try:
            message = f"""
🧠 <b>NEURAL MARKET INSIGHT</b> 🧠

<b>Type:</b> {insight_type}
<b>Insight:</b> {title}
<b>AI Confidence:</b> {confidence:.1%}

{description}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚡ <i>M3 Max Neural Analysis</i>
            """.strip()
            
            return await self.send_message(message, disable_notification=True)
            
        except Exception as e:
            logger.error(f"Error sending market insight: {e}")
            return False
    
    async def send_system_status(
        self, 
        status: str, 
        details: Dict[str, Any]
    ) -> bool:
        """
        Send system status update.
        
        Args:
            status: System status (ONLINE, OFFLINE, ERROR)
            details: Status details
        
        Returns:
            True if successful, False otherwise
        """
        try:
            emoji_map = {
                'ONLINE': '✅',
                'OFFLINE': '🔴',
                'ERROR': '⚠️',
                'MAINTENANCE': '🔧'
            }
            
            emoji = emoji_map.get(status.upper(), '📊')
            
            message = f"""
{emoji} <b>SYSTEM STATUS: {status.upper()}</b> {emoji}

<b>Neural Engine:</b> {details.get('neural_engine', 'Unknown')}
<b>Data Feeds:</b> {details.get('data_feeds', 'Unknown')}
<b>Trading:</b> {details.get('trading_status', 'Unknown')}
<b>Portfolio Value:</b> ${details.get('portfolio_value', 0):,.2f}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🤖 <i>QuantBot AI System</i>
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending system status: {e}")
            return False
    
    def send_message_sync(self, message: str) -> bool:
        """
        Synchronous wrapper for sending messages.
        
        Args:
            message: Message to send
        
        Returns:
            True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.send_message(message))
        except Exception as e:
            logger.error(f"Error in sync message send: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            test_message = "🧠 Neural Trading System - Connection Test ✅"
            return await self.send_message(test_message)
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False