"""
Email notification system for trading alerts and reports.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from loguru import logger

from ..config.settings import settings


class EmailNotifier:
    """
    Email notification system for trading alerts and reports.
    """
    
    def __init__(self):
        self.smtp_server = settings.notifications.email_smtp_server
        self.smtp_port = settings.notifications.email_smtp_port
        self.username = settings.notifications.email_username
        self.password = settings.notifications.email_password
        self.recipients = settings.notifications.email_recipients
        
        if not all([self.smtp_server, self.username, self.password]):
            logger.warning("Email credentials not configured")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("EmailNotifier initialized successfully")
    
    def _create_message(
        self, 
        subject: str, 
        body: str, 
        recipients: Optional[List[str]] = None,
        html_body: Optional[str] = None
    ) -> MIMEMultipart:
        """
        Create email message.
        
        Args:
            subject: Email subject
            body: Plain text body
            recipients: List of recipient emails
            html_body: HTML body (optional)
        
        Returns:
            Email message object
        """
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.username
        message["To"] = ", ".join(recipients or self.recipients)
        
        # Add plain text part
        text_part = MIMEText(body, "plain")
        message.attach(text_part)
        
        # Add HTML part if provided
        if html_body:
            html_part = MIMEText(html_body, "html")
            message.attach(html_part)
        
        return message
    
    def send_email(
        self, 
        subject: str, 
        body: str, 
        recipients: Optional[List[str]] = None,
        html_body: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            body: Plain text body
            recipients: List of recipient emails
            html_body: HTML body (optional)
            attachments: List of file paths to attach
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Email not configured, skipping notification")
            return False
        
        try:
            message = self._create_message(subject, body, recipients, html_body)
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    try:
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {file_path.split("/")[-1]}'
                        )
                        message.attach(part)
                    except Exception as e:
                        logger.warning(f"Failed to attach file {file_path}: {e}")
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                
                text = message.as_string()
                server.sendmail(
                    self.username, 
                    recipients or self.recipients, 
                    text
                )
            
            logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_trade_alert(
        self, 
        symbol: str, 
        action: str, 
        quantity: float, 
        price: float, 
        strategy: str = "Neural AI"
    ) -> bool:
        """
        Send trade execution alert via email.
        
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
            subject = f"🚀 Neural Trade Executed: {action.upper()} {symbol}"
            
            body = f"""
Neural Trading System - Trade Execution Alert

Symbol: {symbol}
Action: {action.upper()}
Quantity: {quantity:,.0f} shares
Price: ${price:.2f}
Total Value: ${quantity * price:,.2f}
Strategy: {strategy}
Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This trade was executed by the Neural AI Trading System powered by M3 Max.

Best regards,
QuantBot AI Neural Trading System
            """.strip()
            
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #0a0a0f; color: #ffffff;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #00ffff; text-align: center;">🚀 Neural Trade Executed</h2>
                    
                    <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; border: 1px solid #00ffff;">
                        <table style="width: 100%; color: #ffffff;">
                            <tr><td><strong>Symbol:</strong></td><td>{symbol}</td></tr>
                            <tr><td><strong>Action:</strong></td><td style="color: {'#00ff88' if action.upper() == 'BUY' else '#ff6b35'};">{action.upper()}</td></tr>
                            <tr><td><strong>Quantity:</strong></td><td>{quantity:,.0f} shares</td></tr>
                            <tr><td><strong>Price:</strong></td><td>${price:.2f}</td></tr>
                            <tr><td><strong>Total Value:</strong></td><td>${quantity * price:,.2f}</td></tr>
                            <tr><td><strong>Strategy:</strong></td><td>{strategy}</td></tr>
                            <tr><td><strong>Time:</strong></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                        </table>
                    </div>
                    
                    <p style="text-align: center; margin-top: 20px; color: #a1a1aa;">
                        🧠 Powered by M3 Max Neural Engine
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self.send_email(subject, body, html_body=html_body)
            
        except Exception as e:
            logger.error(f"Error sending trade alert email: {e}")
            return False
    
    def send_daily_report(
        self, 
        portfolio_summary: Dict[str, Any],
        performance_metrics: Dict[str, float],
        top_positions: List[Dict[str, Any]]
    ) -> bool:
        """
        Send daily portfolio performance report.
        
        Args:
            portfolio_summary: Portfolio summary data
            performance_metrics: Performance metrics
            top_positions: Top performing positions
        
        Returns:
            True if successful, False otherwise
        """
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            subject = f"📊 Daily Neural Trading Report - {date_str}"
            
            # Create plain text report
            body = f"""
Neural Trading System - Daily Performance Report
Date: {date_str}

PORTFOLIO SUMMARY
=================
Total Value: ${portfolio_summary.get('total_value', 0):,.2f}
Daily P&L: ${portfolio_summary.get('daily_pnl', 0):,.2f}
Daily Return: {portfolio_summary.get('daily_return', 0):.2f}%
Total Return: {portfolio_summary.get('total_return', 0):.2f}%

PERFORMANCE METRICS
==================
Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2f}%
Win Rate: {performance_metrics.get('win_rate', 0):.1f}%
Total Trades: {performance_metrics.get('total_trades', 0)}

TOP POSITIONS
=============
            """
            
            for i, position in enumerate(top_positions[:5], 1):
                body += f"{i}. {position.get('symbol', 'N/A')}: ${position.get('value', 0):,.2f} ({position.get('return', 0):+.1f}%)\n"
            
            body += f"""

Generated by QuantBot AI Neural Trading System
Powered by M3 Max Neural Engine
            """
            
            # Create HTML report
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #0a0a0f; color: #ffffff;">
                <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                    <h1 style="color: #00ffff; text-align: center;">📊 Daily Neural Trading Report</h1>
                    <p style="text-align: center; color: #a1a1aa;">{date_str}</p>
                    
                    <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #00ffff;">
                        <h3 style="color: #00ffff;">Portfolio Summary</h3>
                        <table style="width: 100%; color: #ffffff;">
                            <tr><td>Total Value:</td><td style="text-align: right; color: #00ff88;">${portfolio_summary.get('total_value', 0):,.2f}</td></tr>
                            <tr><td>Daily P&L:</td><td style="text-align: right; color: {'#00ff88' if portfolio_summary.get('daily_pnl', 0) >= 0 else '#ff6b35'};">${portfolio_summary.get('daily_pnl', 0):,.2f}</td></tr>
                            <tr><td>Daily Return:</td><td style="text-align: right; color: {'#00ff88' if portfolio_summary.get('daily_return', 0) >= 0 else '#ff6b35'};">{portfolio_summary.get('daily_return', 0):+.2f}%</td></tr>
                            <tr><td>Total Return:</td><td style="text-align: right; color: {'#00ff88' if portfolio_summary.get('total_return', 0) >= 0 else '#ff6b35'};">{portfolio_summary.get('total_return', 0):+.2f}%</td></tr>
                        </table>
                    </div>
                    
                    <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #ff00ff;">
                        <h3 style="color: #ff00ff;">Performance Metrics</h3>
                        <table style="width: 100%; color: #ffffff;">
                            <tr><td>Sharpe Ratio:</td><td style="text-align: right;">{performance_metrics.get('sharpe_ratio', 0):.2f}</td></tr>
                            <tr><td>Max Drawdown:</td><td style="text-align: right;">{performance_metrics.get('max_drawdown', 0):.2f}%</td></tr>
                            <tr><td>Win Rate:</td><td style="text-align: right;">{performance_metrics.get('win_rate', 0):.1f}%</td></tr>
                            <tr><td>Total Trades:</td><td style="text-align: right;">{performance_metrics.get('total_trades', 0)}</td></tr>
                        </table>
                    </div>
                    
                    <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #00ff88;">
                        <h3 style="color: #00ff88;">Top Positions</h3>
                        <table style="width: 100%; color: #ffffff;">
            """
            
            for i, position in enumerate(top_positions[:5], 1):
                return_color = '#00ff88' if position.get('return', 0) >= 0 else '#ff6b35'
                html_body += f"""
                            <tr>
                                <td>{i}.</td>
                                <td>{position.get('symbol', 'N/A')}</td>
                                <td style="text-align: right;">${position.get('value', 0):,.2f}</td>
                                <td style="text-align: right; color: {return_color};">{position.get('return', 0):+.1f}%</td>
                            </tr>
                """
            
            html_body += """
                        </table>
                    </div>
                    
                    <p style="text-align: center; margin-top: 30px; color: #a1a1aa;">
                        🧠 Generated by QuantBot AI Neural Trading System<br>
                        ⚡ Powered by M3 Max Neural Engine
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self.send_email(subject, body, html_body=html_body)
            
        except Exception as e:
            logger.error(f"Error sending daily report email: {e}")
            return False
    
    def send_risk_alert(
        self, 
        alert_type: str, 
        symbol: str, 
        current_value: float, 
        threshold: float, 
        message_details: str = ""
    ) -> bool:
        """
        Send risk management alert via email.
        
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
            subject = f"🚨 Risk Alert: {alert_type} - {symbol}"
            
            body = f"""
NEURAL TRADING SYSTEM - RISK ALERT

Alert Type: {alert_type}
Symbol: {symbol}
Current Value: {current_value:.2f}
Threshold: {threshold:.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Details:
{message_details}

This alert was generated by the Neural Risk Shield system.
Please review your positions and take appropriate action if necessary.

Best regards,
QuantBot AI Risk Management System
            """.strip()
            
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #0a0a0f; color: #ffffff;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #ff6b35; text-align: center;">🚨 Risk Alert</h2>
                    
                    <div style="background: #2a1a1a; padding: 20px; border-radius: 10px; border: 2px solid #ff6b35;">
                        <table style="width: 100%; color: #ffffff;">
                            <tr><td><strong>Alert Type:</strong></td><td style="color: #ff6b35;">{alert_type}</td></tr>
                            <tr><td><strong>Symbol:</strong></td><td>{symbol}</td></tr>
                            <tr><td><strong>Current Value:</strong></td><td>{current_value:.2f}</td></tr>
                            <tr><td><strong>Threshold:</strong></td><td>{threshold:.2f}</td></tr>
                            <tr><td><strong>Time:</strong></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                        </table>
                        
                        <div style="margin-top: 20px; padding: 15px; background: #1a1a1a; border-radius: 5px;">
                            <strong>Details:</strong><br>
                            {message_details}
                        </div>
                    </div>
                    
                    <p style="text-align: center; margin-top: 20px; color: #a1a1aa;">
                        🛡️ Neural Risk Shield Active
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self.send_email(subject, body, html_body=html_body)
            
        except Exception as e:
            logger.error(f"Error sending risk alert email: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test email connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            subject = "🧠 Neural Trading System - Email Test"
            body = "This is a test email from the QuantBot AI Neural Trading System. If you receive this, email notifications are working correctly."
            
            return self.send_email(subject, body)
            
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False