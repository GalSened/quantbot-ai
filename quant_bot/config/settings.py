"""
Configuration settings for the Quantitative Trading Bot.
"""

import os
from typing import List, Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="quantbot", env="POSTGRES_DB")
    postgres_user: str = Field(default="quantbot", env="POSTGRES_USER")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class APISettings(BaseSettings):
    """API keys and external service settings."""
    
    finnhub_api_key: str = Field(default="", env="FINNHUB_API_KEY")
    alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    polygon_api_key: str = Field(default="", env="POLYGON_API_KEY")


class TradingSettings(BaseSettings):
    """Trading and strategy configuration."""
    
    initial_capital: float = Field(default=100000.0, env="INITIAL_CAPITAL")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")  # 10% of portfolio
    transaction_cost: float = Field(default=0.001, env="TRANSACTION_COST")  # 0.1%
    slippage: float = Field(default=0.0005, env="SLIPPAGE")  # 0.05%
    
    # Risk management
    max_drawdown: float = Field(default=0.15, env="MAX_DRAWDOWN")  # 15%
    stop_loss: float = Field(default=0.05, env="STOP_LOSS")  # 5%
    take_profit: float = Field(default=0.10, env="TAKE_PROFIT")  # 10%
    
    # Trading symbols
    symbols: List[str] = Field(default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"])
    
    # Data settings
    data_frequency: str = Field(default="1h", env="DATA_FREQUENCY")  # 1h, 1d, etc.
    lookback_days: int = Field(default=252, env="LOOKBACK_DAYS")  # 1 year


class RLSettings(BaseSettings):
    """Reinforcement Learning configuration."""
    
    algorithm: str = Field(default="PPO", env="RL_ALGORITHM")  # PPO, DDPG, SAC
    learning_rate: float = Field(default=3e-4, env="RL_LEARNING_RATE")
    batch_size: int = Field(default=64, env="RL_BATCH_SIZE")
    n_steps: int = Field(default=2048, env="RL_N_STEPS")
    n_epochs: int = Field(default=10, env="RL_N_EPOCHS")
    
    # Environment settings
    state_dim: int = Field(default=50, env="RL_STATE_DIM")  # Technical indicators + price history
    action_dim: int = Field(default=3, env="RL_ACTION_DIM")  # Buy, Hold, Sell
    reward_scaling: float = Field(default=1.0, env="RL_REWARD_SCALING")
    
    # Training settings
    total_timesteps: int = Field(default=100000, env="RL_TOTAL_TIMESTEPS")
    eval_freq: int = Field(default=5000, env="RL_EVAL_FREQ")
    save_freq: int = Field(default=10000, env="RL_SAVE_FREQ")


class NotificationSettings(BaseSettings):
    """Notification service settings."""
    
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")
    
    email_smtp_server: str = Field(default="smtp.gmail.com", env="EMAIL_SMTP_SERVER")
    email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: str = Field(default="", env="EMAIL_USERNAME")
    email_password: str = Field(default="", env="EMAIL_PASSWORD")
    email_recipients: List[str] = Field(default=[], env="EMAIL_RECIPIENTS")


class DashboardSettings(BaseSettings):
    """Dashboard configuration."""
    
    host: str = Field(default="0.0.0.0", env="DASHBOARD_HOST")
    port: int = Field(default=8501, env="DASHBOARD_PORT")
    debug: bool = Field(default=False, env="DASHBOARD_DEBUG")
    update_interval: int = Field(default=30, env="DASHBOARD_UPDATE_INTERVAL")  # seconds


class Settings:
    """Main settings class combining all configurations."""
    
    def __init__(self):
        self.database = DatabaseSettings()
        self.api = APISettings()
        self.trading = TradingSettings()
        self.rl = RLSettings()
        self.notifications = NotificationSettings()
        self.dashboard = DashboardSettings()
        
        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file: str = os.getenv("LOG_FILE", "quant_bot.log")
        
        # Environment
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"


# Global settings instance
settings = Settings()