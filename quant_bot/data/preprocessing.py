"""
Data preprocessing module for cleaning and preparing market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from loguru import logger

from ..config.settings import settings


class DataPreprocessor:
    """
    Handles data preprocessing including:
    - Data cleaning and validation
    - Technical indicator calculation
    - Sentiment analysis
    - Feature engineering
    - Data normalization
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='forward_fill')
        
        logger.info("DataPreprocessor initialized successfully")
    
    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate market data.
        
        Args:
            data: Raw market data DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Cleaning market data with {len(data)} records")
            
            # Make a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Ensure datetime index
            if 'Datetime' in cleaned_data.columns:
                cleaned_data['Datetime'] = pd.to_datetime(cleaned_data['Datetime'])
                cleaned_data.set_index('Datetime', inplace=True)
            elif not isinstance(cleaned_data.index, pd.DatetimeIndex):
                cleaned_data.index = pd.to_datetime(cleaned_data.index)
            
            # Remove duplicates
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='last')]
            
            # Sort by datetime
            cleaned_data.sort_index(inplace=True)
            
            # Validate OHLCV data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                return pd.DataFrame()
            
            # Remove rows with invalid OHLC relationships
            invalid_mask = (
                (cleaned_data['High'] < cleaned_data['Low']) |
                (cleaned_data['High'] < cleaned_data['Open']) |
                (cleaned_data['High'] < cleaned_data['Close']) |
                (cleaned_data['Low'] > cleaned_data['Open']) |
                (cleaned_data['Low'] > cleaned_data['Close'])
            )
            
            if invalid_mask.any():
                logger.warning(f"Removing {invalid_mask.sum()} rows with invalid OHLC relationships")
                cleaned_data = cleaned_data[~invalid_mask]
            
            # Remove rows with zero or negative prices
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                cleaned_data = cleaned_data[cleaned_data[col] > 0]
            
            # Remove rows with negative volume
            cleaned_data = cleaned_data[cleaned_data['Volume'] >= 0]
            
            # Handle missing values
            if cleaned_data.isnull().any().any():
                logger.info("Handling missing values")
                # Forward fill missing values
                cleaned_data.fillna(method='ffill', inplace=True)
                # If still missing values at the beginning, backward fill
                cleaned_data.fillna(method='bfill', inplace=True)
            
            # Remove extreme outliers (prices that change more than 50% in one period)
            for col in price_columns:
                pct_change = cleaned_data[col].pct_change().abs()
                outlier_mask = pct_change > 0.5
                if outlier_mask.any():
                    logger.warning(f"Removing {outlier_mask.sum()} outliers from {col}")
                    cleaned_data = cleaned_data[~outlier_mask]
            
            logger.info(f"Data cleaning completed. {len(cleaned_data)} records remaining")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning market data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given market data.
        
        Args:
            data: Cleaned market data DataFrame
        
        Returns:
            DataFrame with technical indicators added
        """
        try:
            logger.info("Calculating technical indicators")
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Price-based indicators
            df['SMA_10'] = ta.sma(df['Close'], length=10)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['EMA_12'] = ta.ema(df['Close'], length=12)
            df['EMA_26'] = ta.ema(df['Close'], length=26)
            
            # Bollinger Bands
            bb = ta.bbands(df['Close'], length=20, std=2)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
            
            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # MACD
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
            
            # Stochastic Oscillator
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
            if stoch is not None:
                df = pd.concat([df, stoch], axis=1)
            
            # Williams %R
            df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            
            # Average True Range (ATR)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Volume indicators
            df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # On-Balance Volume
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            
            # Commodity Channel Index
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            
            # Money Flow Index
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
            
            # Price momentum
            df['ROC'] = ta.roc(df['Close'], length=10)  # Rate of Change
            df['Momentum'] = ta.mom(df['Close'], length=10)
            
            # Volatility indicators
            df['Price_Volatility'] = df['Close'].rolling(window=20).std()
            df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
            
            # Support and Resistance levels (simplified)
            df['Resistance'] = df['High'].rolling(window=20).max()
            df['Support'] = df['Low'].rolling(window=20).min()
            
            # Price position within range
            df['Price_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
            
            # Trend indicators
            df['Price_Trend'] = np.where(df['Close'] > df['SMA_20'], 1, 
                                np.where(df['Close'] < df['SMA_20'], -1, 0))
            
            # Volume trend
            df['Volume_Trend'] = np.where(df['Volume'] > df['Volume_SMA'], 1, -1)
            
            logger.info(f"Technical indicators calculated. DataFrame shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    def analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze sentiment from news data.
        
        Args:
            news_data: List of news articles with headlines and summaries
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            if not news_data:
                return {
                    'compound_score': 0.0,
                    'positive_score': 0.0,
                    'negative_score': 0.0,
                    'neutral_score': 0.0,
                    'article_count': 0
                }
            
            logger.info(f"Analyzing sentiment for {len(news_data)} articles")
            
            compound_scores = []
            positive_scores = []
            negative_scores = []
            neutral_scores = []
            
            for article in news_data:
                # Combine headline and summary for analysis
                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                
                if not text.strip():
                    continue
                
                # VADER sentiment analysis
                vader_scores = self.sentiment_analyzer.polarity_scores(text)
                
                # TextBlob sentiment analysis
                blob = TextBlob(text)
                textblob_polarity = blob.sentiment.polarity
                
                # Combine both methods (weighted average)
                combined_compound = (vader_scores['compound'] * 0.7 + textblob_polarity * 0.3)
                
                compound_scores.append(combined_compound)
                positive_scores.append(vader_scores['pos'])
                negative_scores.append(vader_scores['neg'])
                neutral_scores.append(vader_scores['neu'])
            
            if not compound_scores:
                return {
                    'compound_score': 0.0,
                    'positive_score': 0.0,
                    'negative_score': 0.0,
                    'neutral_score': 0.0,
                    'article_count': 0
                }
            
            # Calculate weighted averages (more recent articles have higher weight)
            weights = np.exp(np.linspace(-1, 0, len(compound_scores)))
            weights = weights / weights.sum()
            
            sentiment_summary = {
                'compound_score': np.average(compound_scores, weights=weights),
                'positive_score': np.average(positive_scores, weights=weights),
                'negative_score': np.average(negative_scores, weights=weights),
                'neutral_score': np.average(neutral_scores, weights=weights),
                'article_count': len(compound_scores),
                'sentiment_volatility': np.std(compound_scores),
                'bullish_articles': sum(1 for score in compound_scores if score > 0.1),
                'bearish_articles': sum(1 for score in compound_scores if score < -0.1)
            }
            
            logger.info(f"Sentiment analysis completed. Compound score: {sentiment_summary['compound_score']:.3f}")
            return sentiment_summary
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'compound_score': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0,
                'article_count': 0
            }
    
    def create_features_for_ml(
        self, 
        market_data: pd.DataFrame, 
        sentiment_data: Optional[Dict[str, float]] = None,
        lookback_window: int = 20
    ) -> pd.DataFrame:
        """
        Create features suitable for machine learning models.
        
        Args:
            market_data: DataFrame with market data and technical indicators
            sentiment_data: Optional sentiment analysis results
            lookback_window: Number of periods to look back for features
        
        Returns:
            DataFrame with ML-ready features
        """
        try:
            logger.info("Creating features for machine learning")
            
            df = market_data.copy()
            
            # Price-based features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=lookback_window).std()
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
                df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'Close_Mean_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
                df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'High_Max_{window}'] = df['High'].rolling(window=window).max()
                df[f'Low_Min_{window}'] = df['Low'].rolling(window=window).min()
            
            # Price ratios
            df['Close_to_Open'] = df['Close'] / df['Open']
            df['High_to_Close'] = df['High'] / df['Close']
            df['Low_to_Close'] = df['Low'] / df['Close']
            df['Close_to_SMA20'] = df['Close'] / df['SMA_20']
            
            # Momentum features
            for period in [5, 10, 20]:
                df[f'Price_Change_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
                df[f'Volume_Change_{period}'] = (df['Volume'] - df['Volume'].shift(period)) / df['Volume'].shift(period)
            
            # Volatility features
            df['GARCH_Volatility'] = df['Returns'].rolling(window=20).apply(
                lambda x: np.sqrt(np.mean(x**2)), raw=True
            )
            
            # Market microstructure features
            df['Spread'] = (df['High'] - df['Low']) / df['Close']
            df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
            df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
            df['Body_Size'] = np.abs(df['Close'] - df['Open']) / df['Close']
            
            # Time-based features
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
            
            # Cyclical encoding for time features
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
            
            # Add sentiment features if available
            if sentiment_data:
                for key, value in sentiment_data.items():
                    df[f'Sentiment_{key}'] = value
            
            # Target variable for supervised learning (next period return)
            df['Target_Return'] = df['Returns'].shift(-1)
            df['Target_Direction'] = np.where(df['Target_Return'] > 0, 1, 0)
            
            # Remove rows with NaN values
            initial_rows = len(df)
            df.dropna(inplace=True)
            final_rows = len(df)
            
            logger.info(f"Feature engineering completed. Rows: {initial_rows} -> {final_rows}, Features: {df.shape[1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()
    
    def normalize_features(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str],
        fit_scaler: bool = True
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normalize features for machine learning.
        
        Args:
            data: DataFrame with features
            feature_columns: List of columns to normalize
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            Tuple of (normalized DataFrame, fitted scaler)
        """
        try:
            logger.info(f"Normalizing {len(feature_columns)} features")
            
            df = data.copy()
            
            # Select only the specified feature columns that exist in the DataFrame
            existing_columns = [col for col in feature_columns if col in df.columns]
            missing_columns = [col for col in feature_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for normalization: {missing_columns}")
            
            if not existing_columns:
                logger.error("No valid columns found for normalization")
                return df, self.scaler
            
            # Fit and transform or just transform
            if fit_scaler:
                normalized_values = self.scaler.fit_transform(df[existing_columns])
            else:
                normalized_values = self.scaler.transform(df[existing_columns])
            
            # Replace the original columns with normalized values
            df[existing_columns] = normalized_values
            
            logger.info("Feature normalization completed")
            return df, self.scaler
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return data, self.scaler
    
    def create_sequences_for_lstm(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str],
        target_column: str,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_column: Name of target column
            sequence_length: Length of input sequences
        
        Returns:
            Tuple of (X sequences, y targets)
        """
        try:
            logger.info(f"Creating LSTM sequences with length {sequence_length}")
            
            # Select features and target
            features = data[feature_columns].values
            targets = data[target_column].values
            
            X, y = [], []
            
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
                y.append(targets[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating LSTM sequences: {e}")
            return np.array([]), np.array([])
    
    def detect_regime_changes(self, data: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Detect market regime changes using volatility and trend analysis.
        
        Args:
            data: DataFrame with market data
            window: Rolling window for regime detection
        
        Returns:
            DataFrame with regime indicators added
        """
        try:
            logger.info("Detecting market regime changes")
            
            df = data.copy()
            
            # Calculate rolling volatility and returns
            df['Rolling_Vol'] = df['Returns'].rolling(window=window).std()
            df['Rolling_Return'] = df['Returns'].rolling(window=window).mean()
            
            # Define regime thresholds (can be optimized)
            vol_threshold = df['Rolling_Vol'].quantile(0.7)
            return_threshold = 0.001  # 0.1% daily return
            
            # Classify regimes
            conditions = [
                (df['Rolling_Vol'] > vol_threshold) & (df['Rolling_Return'] > return_threshold),
                (df['Rolling_Vol'] > vol_threshold) & (df['Rolling_Return'] < -return_threshold),
                (df['Rolling_Vol'] > vol_threshold),
                (df['Rolling_Return'] > return_threshold),
                (df['Rolling_Return'] < -return_threshold)
            ]
            
            choices = [
                'Bull_Volatile',    # High vol, positive returns
                'Bear_Volatile',    # High vol, negative returns  
                'High_Volatility',  # High vol, neutral returns
                'Bull_Stable',      # Low vol, positive returns
                'Bear_Stable'       # Low vol, negative returns
            ]
            
            df['Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            # Create regime dummy variables for ML
            regime_dummies = pd.get_dummies(df['Market_Regime'], prefix='Regime')
            df = pd.concat([df, regime_dummies], axis=1)
            
            logger.info("Market regime detection completed")
            return df
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
            return data