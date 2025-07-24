"""
Advanced portfolio management with risk controls and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .broker_sim import BrokerSimulator, OrderSide, OrderType
from ..config.settings import settings


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio monitoring."""
    var_95: float = 0.0
    var_99: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    beta: float = 0.0
    correlation_spy: float = 0.0


@dataclass
class PositionSizing:
    """Position sizing parameters."""
    method: str = "fixed_percent"  # fixed_percent, volatility_target, kelly
    base_size: float = 0.02  # 2% of portfolio
    max_size: float = 0.10   # 10% max position
    volatility_target: float = 0.15  # 15% annual volatility target
    lookback_days: int = 20


class PortfolioManager:
    """
    Advanced portfolio management system with risk controls.
    """
    
    def __init__(
        self,
        broker: BrokerSimulator,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        position_sizing: Optional[PositionSizing] = None
    ):
        self.broker = broker
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_sizing = position_sizing or PositionSizing()
        
        # Portfolio tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.returns_history: List[float] = []
        self.equity_curve: List[float] = [initial_capital]
        
        # Risk management
        self.risk_metrics = RiskMetrics()
        self.stop_losses: Dict[str, float] = {}
        self.take_profits: Dict[str, float] = {}
        
        # Market data
        self.price_history: Dict[str, List[float]] = {}
        self.volatility_estimates: Dict[str, float] = {}
        
        # Performance tracking
        self.benchmark_returns: List[float] = []
        self.last_rebalance: Optional[datetime] = None
        
        logger.info("PortfolioManager initialized")
    
    def update_market_data(self, market_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Update market data and recalculate risk metrics.
        
        Args:
            market_data: Dictionary of symbol -> market data
        """
        try:
            # Update broker with current prices
            for symbol, data in market_data.items():
                price = data.get('price', data.get('close', 0))
                if price > 0:
                    self.broker.update_market_data(symbol, price)
                    
                    # Update price history
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    self.price_history[symbol].append(price)
                    
                    # Keep only recent history
                    if len(self.price_history[symbol]) > 252:  # 1 year of daily data
                        self.price_history[symbol] = self.price_history[symbol][-252:]
                    
                    # Update volatility estimate
                    self._update_volatility_estimate(symbol)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check risk limits
            self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def generate_trade_signals(
        self, 
        signals: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate trade recommendations based on signals and risk management.
        
        Args:
            signals: Dictionary of symbol -> signal data
        
        Returns:
            List of trade recommendations
        """
        try:
            recommendations = []
            current_positions = self.broker.get_positions()
            account_summary = self.broker.get_account_summary()
            
            for symbol, signal_data in signals.items():
                try:
                    signal = signal_data.get('signal', 'hold')
                    confidence = signal_data.get('confidence', 0.0)
                    strength = signal_data.get('strength', 0.0)
                    
                    # Skip low confidence signals
                    if confidence < 0.5:
                        continue
                    
                    current_position = current_positions.get(symbol, {}).get('quantity', 0)
                    current_price = self.broker.current_prices.get(symbol, 0)
                    
                    if current_price <= 0:
                        continue
                    
                    # Generate recommendation based on signal
                    recommendation = self._generate_trade_recommendation(
                        symbol, signal, confidence, strength, 
                        current_position, current_price, account_summary
                    )
                    
                    if recommendation:
                        recommendations.append(recommendation)
                        
                except Exception as e:
                    logger.warning(f"Error processing signal for {symbol}: {e}")
                    continue
            
            # Sort by priority (confidence * strength)
            recommendations.sort(
                key=lambda x: x.get('priority', 0), 
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trade signals: {e}")
            return []
    
    def _generate_trade_recommendation(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        strength: float,
        current_position: float,
        current_price: float,
        account_summary: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate individual trade recommendation."""
        
        try:
            # Calculate position size
            target_position_value = self._calculate_position_size(
                symbol, signal, confidence, strength, current_price
            )
            
            if target_position_value == 0:
                return None
            
            target_shares = target_position_value / current_price
            position_change = target_shares - current_position
            
            # Minimum trade threshold
            min_trade_value = 1000  # $1000 minimum trade
            if abs(position_change * current_price) < min_trade_value:
                return None
            
            # Determine order side and quantity
            if position_change > 0:
                side = OrderSide.BUY
                quantity = abs(position_change)
            else:
                side = OrderSide.SELL
                quantity = abs(position_change)
            
            # Check if we can afford the trade
            if side == OrderSide.BUY:
                required_cash = quantity * current_price * 1.01  # Add buffer for slippage
                if required_cash > account_summary['cash']:
                    # Reduce quantity to fit available cash
                    quantity = account_summary['cash'] / (current_price * 1.01)
                    if quantity * current_price < min_trade_value:
                        return None
            
            # Calculate priority
            priority = confidence * strength
            
            # Set stop loss and take profit levels
            stop_loss_price = self._calculate_stop_loss(symbol, current_price, side)
            take_profit_price = self._calculate_take_profit(symbol, current_price, side)
            
            recommendation = {
                'symbol': symbol,
                'side': side.value,
                'quantity': round(quantity, 2),
                'order_type': OrderType.MARKET.value,
                'current_price': current_price,
                'signal': signal,
                'confidence': confidence,
                'strength': strength,
                'priority': priority,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'reasoning': f"{signal.upper()} signal with {confidence:.1%} confidence",
                'risk_reward_ratio': self._calculate_risk_reward_ratio(
                    current_price, stop_loss_price, take_profit_price, side
                )
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            return None
    
    def execute_trades(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """
        Execute trade recommendations.
        
        Args:
            recommendations: List of trade recommendations
        
        Returns:
            List of order IDs
        """
        try:
            order_ids = []
            account_summary = self.broker.get_account_summary()
            
            # Check overall portfolio risk before executing trades
            if not self._check_portfolio_risk():
                logger.warning("Portfolio risk limits exceeded, skipping trades")
                return order_ids
            
            for rec in recommendations:
                try:
                    # Final risk check for individual trade
                    if not self._check_trade_risk(rec):
                        logger.info(f"Skipping {rec['symbol']} trade due to risk limits")
                        continue
                    
                    # Place order
                    order_id = self.broker.place_order(
                        symbol=rec['symbol'],
                        side=rec['side'],
                        quantity=rec['quantity'],
                        order_type=rec['order_type']
                    )
                    
                    order_ids.append(order_id)
                    
                    # Set stop loss and take profit orders
                    if rec.get('stop_loss'):
                        self._set_stop_loss(rec['symbol'], rec['stop_loss'])
                    
                    if rec.get('take_profit'):
                        self._set_take_profit(rec['symbol'], rec['take_profit'])
                    
                    logger.info(f"Executed trade: {rec['side']} {rec['quantity']} {rec['symbol']}")
                    
                except Exception as e:
                    logger.error(f"Error executing trade for {rec['symbol']}: {e}")
                    continue
            
            return order_ids
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return []
    
    def _calculate_position_size(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        strength: float,
        current_price: float
    ) -> float:
        """Calculate optimal position size based on sizing method."""
        
        try:
            account_summary = self.broker.get_account_summary()
            portfolio_value = account_summary['equity']
            
            if self.position_sizing.method == "fixed_percent":
                # Fixed percentage of portfolio
                base_allocation = portfolio_value * self.position_sizing.base_size
                # Adjust by confidence and strength
                adjusted_allocation = base_allocation * confidence * (1 + strength)
                
            elif self.position_sizing.method == "volatility_target":
                # Volatility targeting
                symbol_volatility = self.volatility_estimates.get(symbol, 0.2)
                target_volatility = self.position_sizing.volatility_target
                
                # Scale position size inversely with volatility
                volatility_scalar = target_volatility / symbol_volatility
                base_allocation = portfolio_value * self.position_sizing.base_size
                adjusted_allocation = base_allocation * volatility_scalar * confidence
                
            elif self.position_sizing.method == "kelly":
                # Kelly criterion (simplified)
                win_rate = confidence
                avg_win = strength * 0.1  # Assume 10% average win
                avg_loss = 0.05  # Assume 5% average loss
                
                if win_rate > 0 and avg_win > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                    adjusted_allocation = portfolio_value * kelly_fraction
                else:
                    adjusted_allocation = 0
                    
            else:
                adjusted_allocation = portfolio_value * self.position_sizing.base_size
            
            # Apply maximum position size limit
            max_allocation = portfolio_value * self.position_sizing.max_size
            adjusted_allocation = min(adjusted_allocation, max_allocation)
            
            # Don't trade if signal is hold or allocation is too small
            if signal == 'hold' or adjusted_allocation < 1000:
                return 0
            
            return adjusted_allocation
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def _calculate_stop_loss(self, symbol: str, current_price: float, side: OrderSide) -> Optional[float]:
        """Calculate stop loss price."""
        
        try:
            # Use ATR-based stop loss
            volatility = self.volatility_estimates.get(symbol, 0.02)  # 2% default
            atr_multiplier = 2.0
            
            if side == OrderSide.BUY:
                stop_loss = current_price * (1 - volatility * atr_multiplier)
            else:
                stop_loss = current_price * (1 + volatility * atr_multiplier)
            
            return round(stop_loss, 2)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {e}")
            return None
    
    def _calculate_take_profit(self, symbol: str, current_price: float, side: OrderSide) -> Optional[float]:
        """Calculate take profit price."""
        
        try:
            # Use 2:1 risk-reward ratio
            volatility = self.volatility_estimates.get(symbol, 0.02)
            profit_multiplier = 4.0  # 2x the stop loss distance
            
            if side == OrderSide.BUY:
                take_profit = current_price * (1 + volatility * profit_multiplier)
            else:
                take_profit = current_price * (1 - volatility * profit_multiplier)
            
            return round(take_profit, 2)
            
        except Exception as e:
            logger.error(f"Error calculating take profit for {symbol}: {e}")
            return None
    
    def _calculate_risk_reward_ratio(
        self, 
        entry_price: float, 
        stop_loss: Optional[float], 
        take_profit: Optional[float], 
        side: OrderSide
    ) -> float:
        """Calculate risk-reward ratio."""
        
        if not stop_loss or not take_profit:
            return 0.0
        
        if side == OrderSide.BUY:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        if risk <= 0:
            return 0.0
        
        return reward / risk
    
    def _set_stop_loss(self, symbol: str, stop_price: float) -> None:
        """Set stop loss for a position."""
        self.stop_losses[symbol] = stop_price
    
    def _set_take_profit(self, symbol: str, profit_price: float) -> None:
        """Set take profit for a position."""
        self.take_profits[symbol] = profit_price
    
    def _update_volatility_estimate(self, symbol: str) -> None:
        """Update volatility estimate for a symbol."""
        
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return
            
            prices = np.array(self.price_history[symbol][-30:])  # Last 30 periods
            returns = np.diff(np.log(prices))
            
            if len(returns) > 0:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                self.volatility_estimates[symbol] = volatility
                
        except Exception as e:
            logger.error(f"Error updating volatility for {symbol}: {e}")
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics."""
        
        try:
            account_summary = self.broker.get_account_summary()
            current_equity = account_summary['equity']
            
            # Update equity curve
            self.equity_curve.append(current_equity)
            
            # Calculate returns
            if len(self.equity_curve) > 1:
                period_return = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
                self.returns_history.append(period_return)
                
                # Keep only recent history
                if len(self.returns_history) > 252:
                    self.returns_history = self.returns_history[-252:]
                    self.equity_curve = self.equity_curve[-252:]
            
            # Update risk metrics
            self._calculate_risk_metrics()
            
            # Record portfolio snapshot
            positions = self.broker.get_positions()
            portfolio_snapshot = {
                'timestamp': datetime.now(),
                'equity': current_equity,
                'cash': account_summary['cash'],
                'total_pnl': account_summary.get('total_pnl', 0),
                'num_positions': len(positions),
                'positions': positions.copy(),
                'risk_metrics': {
                    'var_95': self.risk_metrics.var_95,
                    'max_drawdown': self.risk_metrics.max_drawdown,
                    'volatility': self.risk_metrics.volatility,
                    'sharpe_ratio': self.risk_metrics.sharpe_ratio
                }
            }
            
            self.portfolio_history.append(portfolio_snapshot)
            
            # Keep only recent history
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate portfolio risk metrics."""
        
        try:
            if len(self.returns_history) < 10:
                return
            
            returns = np.array(self.returns_history)
            
            # Value at Risk
            self.risk_metrics.var_95 = np.percentile(returns, 5)
            self.risk_metrics.var_99 = np.percentile(returns, 1)
            
            # Volatility
            self.risk_metrics.volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe Ratio (assuming 0% risk-free rate)
            mean_return = np.mean(returns)
            if self.risk_metrics.volatility > 0:
                self.risk_metrics.sharpe_ratio = (mean_return * 252) / self.risk_metrics.volatility
            
            # Maximum Drawdown
            equity_array = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak
            self.risk_metrics.max_drawdown = np.min(drawdown)
            self.risk_metrics.current_drawdown = drawdown[-1]
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
    
    def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        
        try:
            # Check maximum drawdown
            if abs(self.risk_metrics.current_drawdown) > settings.trading.max_drawdown:
                logger.warning(f"Maximum drawdown exceeded: {self.risk_metrics.current_drawdown:.2%}")
                self._emergency_liquidation()
            
            # Check individual position stop losses
            positions = self.broker.get_positions()
            for symbol, position_data in positions.items():
                current_price = self.broker.current_prices.get(symbol, 0)
                if current_price <= 0:
                    continue
                
                quantity = position_data['quantity']
                if quantity == 0:
                    continue
                
                # Check stop loss
                if symbol in self.stop_losses:
                    stop_price = self.stop_losses[symbol]
                    
                    if quantity > 0 and current_price <= stop_price:
                        # Long position hit stop loss
                        self._execute_stop_loss(symbol, quantity)
                    elif quantity < 0 and current_price >= stop_price:
                        # Short position hit stop loss
                        self._execute_stop_loss(symbol, abs(quantity))
                
                # Check take profit
                if symbol in self.take_profits:
                    profit_price = self.take_profits[symbol]
                    
                    if quantity > 0 and current_price >= profit_price:
                        # Long position hit take profit
                        self._execute_take_profit(symbol, quantity)
                    elif quantity < 0 and current_price <= profit_price:
                        # Short position hit take profit
                        self._execute_take_profit(symbol, abs(quantity))
                        
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def _check_portfolio_risk(self) -> bool:
        """Check if portfolio risk is within acceptable limits."""
        
        try:
            # Check current drawdown
            if abs(self.risk_metrics.current_drawdown) > settings.trading.max_drawdown * 0.8:
                return False
            
            # Check number of positions
            positions = self.broker.get_positions()
            if len(positions) >= self.max_positions:
                return False
            
            # Check volatility
            if self.risk_metrics.volatility > 0.5:  # 50% annual volatility limit
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return False
    
    def _check_trade_risk(self, recommendation: Dict[str, Any]) -> bool:
        """Check if individual trade meets risk criteria."""
        
        try:
            # Check risk-reward ratio
            risk_reward = recommendation.get('risk_reward_ratio', 0)
            if risk_reward < 1.5:  # Minimum 1.5:1 risk-reward
                return False
            
            # Check confidence threshold
            confidence = recommendation.get('confidence', 0)
            if confidence < 0.6:  # Minimum 60% confidence
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade risk: {e}")
            return False
    
    def _execute_stop_loss(self, symbol: str, quantity: float) -> None:
        """Execute stop loss order."""
        
        try:
            order_id = self.broker.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            logger.warning(f"Stop loss executed for {symbol}: {quantity} shares")
            
            # Remove stop loss
            if symbol in self.stop_losses:
                del self.stop_losses[symbol]
                
        except Exception as e:
            logger.error(f"Error executing stop loss for {symbol}: {e}")
    
    def _execute_take_profit(self, symbol: str, quantity: float) -> None:
        """Execute take profit order."""
        
        try:
            order_id = self.broker.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            logger.info(f"Take profit executed for {symbol}: {quantity} shares")
            
            # Remove take profit
            if symbol in self.take_profits:
                del self.take_profits[symbol]
                
        except Exception as e:
            logger.error(f"Error executing take profit for {symbol}: {e}")
    
    def _emergency_liquidation(self) -> None:
        """Emergency liquidation of all positions."""
        
        try:
            logger.critical("Emergency liquidation triggered!")
            
            positions = self.broker.get_positions()
            for symbol, position_data in positions.items():
                quantity = position_data['quantity']
                if quantity != 0:
                    self.broker.place_order(
                        symbol=symbol,
                        side=OrderSide.SELL if quantity > 0 else OrderSide.BUY,
                        quantity=abs(quantity),
                        order_type=OrderType.MARKET
                    )
            
            # Clear all stop losses and take profits
            self.stop_losses.clear()
            self.take_profits.clear()
            
        except Exception as e:
            logger.error(f"Error during emergency liquidation: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        
        try:
            account_summary = self.broker.get_account_summary()
            positions = self.broker.get_positions()
            performance_metrics = self.broker.get_performance_metrics()
            
            return {
                'account': account_summary,
                'positions': positions,
                'performance': performance_metrics,
                'risk_metrics': {
                    'var_95': self.risk_metrics.var_95,
                    'var_99': self.risk_metrics.var_99,
                    'max_drawdown': self.risk_metrics.max_drawdown,
                    'current_drawdown': self.risk_metrics.current_drawdown,
                    'volatility': self.risk_metrics.volatility,
                    'sharpe_ratio': self.risk_metrics.sharpe_ratio
                },
                'stop_losses': self.stop_losses.copy(),
                'take_profits': self.take_profits.copy(),
                'num_trades_today': len([t for t in self.broker.get_trade_history() 
                                       if datetime.fromisoformat(t['timestamp']).date() == datetime.now().date()])
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def export_performance_data(self) -> pd.DataFrame:
        """Export portfolio performance data as DataFrame."""
        
        try:
            if not self.portfolio_history:
                return pd.DataFrame()
            
            data = []
            for snapshot in self.portfolio_history:
                row = {
                    'timestamp': snapshot['timestamp'],
                    'equity': snapshot['equity'],
                    'cash': snapshot['cash'],
                    'total_pnl': snapshot['total_pnl'],
                    'num_positions': snapshot['num_positions'],
                    'var_95': snapshot['risk_metrics']['var_95'],
                    'max_drawdown': snapshot['risk_metrics']['max_drawdown'],
                    'volatility': snapshot['risk_metrics']['volatility'],
                    'sharpe_ratio': snapshot['risk_metrics']['sharpe_ratio']
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return pd.DataFrame()