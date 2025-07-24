"""
Trading environment for reinforcement learning.
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from ..config.settings import settings


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    
    State: Technical indicators + price history + sentiment
    Actions: 0=Sell, 1=Hold, 2=Buy
    Reward: Portfolio return adjusted for risk
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        lookback_window: int = 20
    ):
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        
        # Remove NaN values
        self.data.dropna(inplace=True)
        
        if len(self.data) < lookback_window + 1:
            raise ValueError(f"Data too short. Need at least {lookback_window + 1} rows")
        
        # Prepare features
        self.feature_columns = self._select_features()
        self.features = self.data[self.feature_columns].values
        self.prices = self.data['Close'].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(self.data) - lookback_window - 1
        
        # Portfolio state
        self.balance = initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.portfolio_value = initial_balance
        self.trades = []
        self.returns = []
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=Sell, 1=Hold, 2=Buy
        
        # Observation space: features + portfolio state
        obs_dim = len(self.feature_columns) * lookback_window + 3  # +3 for position, balance ratio, portfolio value ratio
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        logger.info(f"TradingEnvironment initialized with {len(self.data)} steps, {len(self.feature_columns)} features")
    
    def _select_features(self) -> List[str]:
        """Select relevant features for the RL model."""
        
        # Technical indicators
        technical_features = [
            'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'SMA_20', 'EMA_12', 'EMA_26', 'ATR_14', 'Volume_Ratio_20',
            'WILLR_14', 'CCI_14', 'MFI_14', 'OBV', 'VWAP'
        ]
        
        # Price-based features
        price_features = [
            'Returns', 'Volatility', 'Close_to_SMA20', 'Price_Position_20',
            'High_Low_Pct', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow'
        ]
        
        # Sentiment features (if available)
        sentiment_features = [
            'Sentiment_overall_sentiment', 'Sentiment_confidence',
            'Sentiment_financial_sentiment'
        ]
        
        # Select features that exist in the data
        all_features = technical_features + price_features + sentiment_features
        available_features = [f for f in all_features if f in self.data.columns]
        
        # If no predefined features available, use numeric columns
        if not available_features:
            available_features = [col for col in self.data.columns 
                                if self.data[col].dtype in ['float64', 'int64'] 
                                and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"Selected {len(available_features)} features for RL environment")
        return available_features[:50]  # Limit to 50 features
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.returns = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Get current and next prices
        current_price = self.prices[self.current_step + self.lookback_window]
        next_price = self.prices[self.current_step + self.lookback_window + 1]
        
        # Execute action
        reward = self._execute_action(action, current_price, next_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps) or (self.portfolio_value <= 0.1 * self.initial_balance)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def _execute_action(self, action: int, current_price: float, next_price: float) -> float:
        """Execute trading action and calculate reward."""
        
        # Map action to position change
        if action == 0:  # Sell
            target_position = -self.max_position
        elif action == 1:  # Hold
            target_position = self.position
        else:  # Buy
            target_position = self.max_position
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Execute trade if position changes
        if abs(position_change) > 0.01:  # Minimum trade threshold
            trade_value = abs(position_change) * current_price * (self.balance / current_price)
            transaction_cost = trade_value * self.transaction_cost
            
            # Update balance and position
            self.balance -= transaction_cost
            self.position = target_position
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'price': current_price,
                'position_change': position_change,
                'transaction_cost': transaction_cost
            })
        
        # Calculate portfolio value change
        price_return = (next_price - current_price) / current_price
        position_return = self.position * price_return
        
        # Update portfolio value
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.balance + self.position * next_price * (self.balance / current_price)
        
        # Calculate reward
        portfolio_return = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        self.returns.append(portfolio_return)
        
        # Risk-adjusted reward (Sharpe ratio approximation)
        if len(self.returns) > 10:
            returns_array = np.array(self.returns[-20:])  # Last 20 returns
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
                reward = sharpe_ratio * 100  # Scale for RL
            else:
                reward = mean_return * 100
        else:
            reward = portfolio_return * 100
        
        # Penalty for excessive trading
        if len(self.trades) > 1:
            recent_trades = len([t for t in self.trades[-10:] if t['step'] > self.current_step - 5])
            if recent_trades > 3:
                reward -= 0.1 * recent_trades
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        
        if self.current_step + self.lookback_window >= len(self.features):
            # Return last valid observation
            start_idx = len(self.features) - self.lookback_window
            end_idx = len(self.features)
        else:
            start_idx = self.current_step
            end_idx = self.current_step + self.lookback_window
        
        # Get feature window
        feature_window = self.features[start_idx:end_idx].flatten()
        
        # Add portfolio state
        current_price = self.prices[min(self.current_step + self.lookback_window, len(self.prices) - 1)]
        portfolio_state = np.array([
            self.position,
            self.balance / self.initial_balance,
            self.portfolio_value / self.initial_balance
        ])
        
        observation = np.concatenate([feature_window, portfolio_state]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        
        current_price = self.prices[min(self.current_step + self.lookback_window, len(self.prices) - 1)]
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'current_price': current_price,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'num_trades': len(self.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns."""
        
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming hourly data)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 24)  # 252 trading days, 24 hours
        
        return sharpe_ratio
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment (for debugging)."""
        
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, Portfolio Value: ${info['portfolio_value']:.2f}, "
                  f"Position: {info['position']:.2f}, Return: {info['total_return']:.2%}, "
                  f"Sharpe: {info['sharpe_ratio']:.2f}")
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio performance history."""
        
        history = []
        for i, trade in enumerate(self.trades):
            history.append({
                'step': trade['step'],
                'action': trade['action'],
                'price': trade['price'],
                'position': trade.get('position_after', 0),
                'portfolio_value': self.portfolio_value if i == len(self.trades) - 1 else None
            })
        
        return pd.DataFrame(history)