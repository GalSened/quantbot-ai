"""
Data acquisition module for fetching market data and news sentiment.
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import redis
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import finnhub
from loguru import logger

from ..config.settings import settings


class DataAcquisition:
    """
    Handles data acquisition from multiple sources including:
    - Yahoo Finance for historical and real-time market data
    - Finnhub for news and sentiment data
    - Alpaca for live trading data
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.database.redis_host,
            port=settings.database.redis_port,
            db=settings.database.redis_db,
            decode_responses=True
        )
        
        self.engine = create_engine(settings.database.postgres_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize API clients
        self.finnhub_client = finnhub.Client(api_key=settings.api.finnhub_api_key)
        
        logger.info("DataAcquisition initialized successfully")
    
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        period: str = "1y",
        interval: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical market data for given symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            Dictionary mapping symbols to their historical data DataFrames
        """
        historical_data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"historical:{symbol}:{period}:{interval}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    logger.info(f"Loading cached historical data for {symbol}")
                    historical_data[symbol] = pd.read_json(cached_data)
                    continue
                
                logger.info(f"Fetching historical data for {symbol}")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Add symbol column
                    data['Symbol'] = symbol
                    data.reset_index(inplace=True)
                    
                    # Cache the data (expire in 1 hour)
                    self.redis_client.setex(
                        cache_key, 
                        3600, 
                        data.to_json(date_format='iso')
                    )
                    
                    historical_data[symbol] = data
                    logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                continue
        
        return historical_data
    
    async def fetch_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch real-time market data for given symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbols to their real-time data
        """
        real_time_data = {}
        
        try:
            # Fetch data for all symbols at once
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    # Get the most recent price data
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        
                        real_time_data[symbol] = {
                            'symbol': symbol,
                            'price': latest['Close'],
                            'volume': latest['Volume'],
                            'high': latest['High'],
                            'low': latest['Low'],
                            'open': latest['Open'],
                            'timestamp': latest.name.isoformat(),
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'dividend_yield': info.get('dividendYield', 0)
                        }
                        
                        # Cache real-time data (expire in 1 minute)
                        cache_key = f"realtime:{symbol}"
                        self.redis_client.setex(
                            cache_key, 
                            60, 
                            json.dumps(real_time_data[symbol])
                        )
                        
                except Exception as e:
                    logger.error(f"Error fetching real-time data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in fetch_real_time_data: {e}")
        
        return real_time_data
    
    async def fetch_news_sentiment(
        self, 
        symbols: List[str], 
        days_back: int = 7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch news and sentiment data for given symbols.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back for news
        
        Returns:
            Dictionary mapping symbols to their news and sentiment data
        """
        news_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"news:{symbol}:{days_back}"
                cached_news = self.redis_client.get(cache_key)
                
                if cached_news:
                    logger.info(f"Loading cached news data for {symbol}")
                    news_data[symbol] = json.loads(cached_news)
                    continue
                
                logger.info(f"Fetching news sentiment for {symbol}")
                
                # Fetch news from Finnhub
                news = self.finnhub_client.company_news(
                    symbol, 
                    _from=start_date.strftime('%Y-%m-%d'), 
                    to=end_date.strftime('%Y-%m-%d')
                )
                
                processed_news = []
                for article in news[:20]:  # Limit to 20 most recent articles
                    processed_news.append({
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'source': article.get('source', ''),
                        'sentiment_score': 0.0  # Will be calculated in preprocessing
                    })
                
                news_data[symbol] = processed_news
                
                # Cache news data (expire in 4 hours)
                self.redis_client.setex(
                    cache_key, 
                    14400, 
                    json.dumps(processed_news)
                )
                
                logger.info(f"Successfully fetched {len(processed_news)} news articles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                news_data[symbol] = []
                continue
        
        return news_data
    
    def save_to_database(self, data: pd.DataFrame, table_name: str) -> bool:
        """
        Save data to PostgreSQL database.
        
        Args:
            data: DataFrame to save
            table_name: Name of the database table
        
        Returns:
            True if successful, False otherwise
        """
        try:
            data.to_sql(
                table_name, 
                self.engine, 
                if_exists='append', 
                index=False,
                method='multi'
            )
            logger.info(f"Successfully saved {len(data)} records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            return False
    
    def load_from_database(
        self, 
        table_name: str, 
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data from PostgreSQL database.
        
        Args:
            table_name: Name of the database table
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with the requested data
        """
        try:
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if start_date:
                conditions.append(f"datetime >= '{start_date.isoformat()}'")
            if end_date:
                conditions.append(f"datetime <= '{end_date.isoformat()}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY datetime"
            
            data = pd.read_sql(query, self.engine)
            logger.info(f"Successfully loaded {len(data)} records from {table_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame()
    
    async def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status and trading hours.
        
        Returns:
            Dictionary with market status information
        """
        try:
            # Use a major index to determine market status
            spy = yf.Ticker("SPY")
            info = spy.info
            
            # Get recent trading data to determine if market is open
            recent_data = spy.history(period="1d", interval="1m")
            
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_market_hours = market_open <= now <= market_close
            is_weekday = now.weekday() < 5
            
            # Check if we have recent data (within last 5 minutes)
            has_recent_data = False
            if not recent_data.empty:
                last_update = recent_data.index[-1].tz_localize(None)
                has_recent_data = (now - last_update).total_seconds() < 300
            
            market_status = {
                'is_open': is_market_hours and is_weekday and has_recent_data,
                'is_market_hours': is_market_hours,
                'is_weekday': is_weekday,
                'current_time': now.isoformat(),
                'market_open': market_open.isoformat(),
                'market_close': market_close.isoformat(),
                'last_data_update': recent_data.index[-1].isoformat() if not recent_data.empty else None
            }
            
            return market_status
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_open': False,
                'error': str(e)
            }
    
    async def cleanup_old_cache(self, days_old: int = 7) -> None:
        """
        Clean up old cached data.
        
        Args:
            days_old: Remove cache entries older than this many days
        """
        try:
            # This is a simplified cleanup - in production, you'd want more sophisticated cache management
            pattern = "historical:*"
            keys = self.redis_client.keys(pattern)
            
            deleted_count = 0
            for key in keys:
                # In a real implementation, you'd check the timestamp of cached data
                # For now, we'll just clean up based on TTL
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    self.redis_client.delete(key)
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old cache entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")