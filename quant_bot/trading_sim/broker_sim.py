"""
Realistic broker simulation for paper trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dataclasses import dataclass, field
from loguru import logger
import asyncio
import random

from ..config.settings import settings


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Account:
    """Account data structure."""
    cash: float = 100000.0
    equity: float = 100000.0
    buying_power: float = 100000.0
    maintenance_margin: float = 0.0
    day_trades: int = 0
    positions: Dict[str, Position] = field(default_factory=dict)


class BrokerSimulator:
    """
    Realistic broker simulation with order execution, slippage, and latency.
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 0.0,
        commission_percent: float = 0.001,
        slippage_model: str = "linear",
        max_slippage: float = 0.005,
        execution_delay_ms: Tuple[int, int] = (10, 100)
    ):
        self.account = Account(
            cash=initial_cash,
            equity=initial_cash,
            buying_power=initial_cash
        )
        
        self.commission_per_trade = commission_per_trade
        self.commission_percent = commission_percent
        self.slippage_model = slippage_model
        self.max_slippage = max_slippage
        self.execution_delay_ms = execution_delay_ms
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.bid_ask_spreads: Dict[str, Tuple[float, float]] = {}
        
        # Risk management
        self.max_position_size = settings.trading.max_position_size
        self.max_drawdown = settings.trading.max_drawdown
        self.stop_loss = settings.trading.stop_loss
        
        logger.info("BrokerSimulator initialized with ${:,.2f} initial cash".format(initial_cash))
    
    def update_market_data(
        self, 
        symbol: str, 
        price: float, 
        bid: Optional[float] = None, 
        ask: Optional[float] = None
    ) -> None:
        """
        Update market data for a symbol.
        
        Args:
            symbol: Stock symbol
            price: Current market price
            bid: Bid price (optional)
            ask: Ask price (optional)
        """
        self.current_prices[symbol] = price
        
        # Estimate bid-ask spread if not provided
        if bid is None or ask is None:
            spread = price * 0.001  # 0.1% spread estimate
            bid = price - spread / 2
            ask = price + spread / 2
        
        self.bid_ask_spreads[symbol] = (bid, ask)
        
        # Update position market values
        if symbol in self.account.positions:
            position = self.account.positions[symbol]
            position.market_value = position.quantity * price
            position.unrealized_pnl = (price - position.avg_price) * position.quantity
        
        # Update account equity
        self._update_account_equity()
    
    def place_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        order_type: Union[OrderType, str] = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """
        Place a trading order.
        
        Args:
            symbol: Stock symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
        
        Returns:
            Order ID
        """
        try:
            # Convert string enums
            if isinstance(side, str):
                side = OrderSide(side.lower())
            if isinstance(order_type, str):
                order_type = OrderType(order_type.lower())
            
            # Validate order
            if not self._validate_order(symbol, side, quantity, order_type, price):
                raise ValueError("Order validation failed")
            
            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            self.orders[order.id] = order
            
            # Process order immediately for market orders
            if order_type == OrderType.MARKET:
                asyncio.create_task(self._execute_order_async(order.id))
            
            logger.info(f"Order placed: {order.id} - {side.value} {quantity} {symbol}")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
                return False
            
            order.status = OrderStatus.CANCELLED
            self.order_history.append(order)
            del self.orders[order_id]
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def _execute_order_async(self, order_id: str) -> None:
        """Execute order asynchronously with realistic delay."""
        
        # Simulate execution delay
        delay_ms = random.randint(*self.execution_delay_ms)
        await asyncio.sleep(delay_ms / 1000.0)
        
        self._execute_order(order_id)
    
    def _execute_order(self, order_id: str) -> None:
        """Execute a trading order."""
        
        try:
            if order_id not in self.orders:
                return
            
            order = self.orders[order_id]
            
            if order.symbol not in self.current_prices:
                order.status = OrderStatus.REJECTED
                logger.warning(f"No market data for {order.symbol}")
                return
            
            current_price = self.current_prices[order.symbol]
            
            # Determine execution price based on order type
            execution_price = self._get_execution_price(order, current_price)
            
            if execution_price is None:
                return  # Order not executable at current price
            
            # Apply slippage
            slippage = self._calculate_slippage(order, execution_price)
            if order.side == OrderSide.BUY:
                execution_price += slippage
            else:
                execution_price -= slippage
            
            # Calculate commission
            commission = self._calculate_commission(order.quantity, execution_price)
            
            # Check if order can be filled
            if not self._can_fill_order(order, execution_price, commission):
                order.status = OrderStatus.REJECTED
                logger.warning(f"Insufficient funds for order {order_id}")
                return
            
            # Execute the trade
            self._fill_order(order, execution_price, commission, slippage)
            
            logger.info(f"Order executed: {order_id} at ${execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing order {order_id}: {e}")
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.REJECTED
    
    def _get_execution_price(self, order: Order, current_price: float) -> Optional[float]:
        """Determine execution price based on order type."""
        
        if order.order_type == OrderType.MARKET:
            return current_price
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                return min(order.price, current_price)
            elif order.side == OrderSide.SELL and current_price >= order.price:
                return max(order.price, current_price)
            return None
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                return current_price
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                return current_price
            return None
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop triggered, now check limit
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                if current_price <= order.price:
                    return min(order.price, current_price)
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                if current_price >= order.price:
                    return max(order.price, current_price)
            return None
        
        return current_price
    
    def _calculate_slippage(self, order: Order, execution_price: float) -> float:
        """Calculate realistic slippage based on order size and market conditions."""
        
        if self.slippage_model == "linear":
            # Linear slippage model based on order size
            base_slippage = execution_price * 0.0001  # 0.01% base slippage
            size_factor = min(order.quantity / 1000, 1.0)  # Scale with order size
            slippage = base_slippage * (1 + size_factor)
            
        elif self.slippage_model == "square_root":
            # Square root model (more realistic for large orders)
            base_slippage = execution_price * 0.0001
            size_factor = np.sqrt(order.quantity / 100)
            slippage = base_slippage * size_factor
            
        else:  # Fixed slippage
            slippage = execution_price * 0.0005
        
        # Cap slippage at maximum
        max_slippage_amount = execution_price * self.max_slippage
        slippage = min(slippage, max_slippage_amount)
        
        return slippage
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission."""
        
        trade_value = quantity * price
        commission = self.commission_per_trade + (trade_value * self.commission_percent)
        
        return commission
    
    def _can_fill_order(self, order: Order, execution_price: float, commission: float) -> bool:
        """Check if order can be filled given current account state."""
        
        if order.side == OrderSide.BUY:
            required_cash = (order.quantity * execution_price) + commission
            return self.account.cash >= required_cash
        
        else:  # SELL
            if order.symbol in self.account.positions:
                available_shares = self.account.positions[order.symbol].quantity
                return available_shares >= order.quantity
            return False
    
    def _fill_order(self, order: Order, execution_price: float, commission: float, slippage: float) -> None:
        """Fill the order and update account state."""
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.filled_at = datetime.now()
        order.commission = commission
        order.slippage = slippage
        
        # Update account and positions
        if order.side == OrderSide.BUY:
            self._process_buy_order(order, execution_price, commission)
        else:
            self._process_sell_order(order, execution_price, commission)
        
        # Move to history
        self.order_history.append(order)
        del self.orders[order.id]
        
        # Record trade
        self.trade_history.append({
            'timestamp': order.filled_at,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'slippage': slippage,
            'order_id': order.id
        })
        
        # Update account equity
        self._update_account_equity()
    
    def _process_buy_order(self, order: Order, execution_price: float, commission: float) -> None:
        """Process a buy order."""
        
        total_cost = (order.quantity * execution_price) + commission
        self.account.cash -= total_cost
        
        # Update position
        if order.symbol not in self.account.positions:
            self.account.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                avg_price=execution_price
            )
        else:
            position = self.account.positions[order.symbol]
            total_quantity = position.quantity + order.quantity
            total_cost_basis = (position.quantity * position.avg_price) + (order.quantity * execution_price)
            
            position.quantity = total_quantity
            position.avg_price = total_cost_basis / total_quantity if total_quantity > 0 else 0
        
        # Update position market value
        current_price = self.current_prices.get(order.symbol, execution_price)
        position = self.account.positions[order.symbol]
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def _process_sell_order(self, order: Order, execution_price: float, commission: float) -> None:
        """Process a sell order."""
        
        if order.symbol not in self.account.positions:
            logger.error(f"No position found for {order.symbol}")
            return
        
        position = self.account.positions[order.symbol]
        
        # Calculate realized P&L
        realized_pnl = (execution_price - position.avg_price) * order.quantity
        position.realized_pnl += realized_pnl
        
        # Update cash
        proceeds = (order.quantity * execution_price) - commission
        self.account.cash += proceeds
        
        # Update position
        position.quantity -= order.quantity
        
        if position.quantity <= 0:
            # Close position
            del self.account.positions[order.symbol]
        else:
            # Update market value and unrealized P&L
            current_price = self.current_prices.get(order.symbol, execution_price)
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def _update_account_equity(self) -> None:
        """Update account equity based on current positions."""
        
        total_market_value = sum(pos.market_value for pos in self.account.positions.values())
        self.account.equity = self.account.cash + total_market_value
        self.account.buying_power = self.account.cash  # Simplified (no margin)
    
    def _validate_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        quantity: float, 
        order_type: OrderType, 
        price: Optional[float]
    ) -> bool:
        """Validate order parameters."""
        
        if quantity <= 0:
            logger.error("Order quantity must be positive")
            return False
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            logger.error("Limit orders require a price")
            return False
        
        if symbol not in self.current_prices:
            logger.error(f"No market data available for {symbol}")
            return False
        
        # Check position size limits
        current_position = self.account.positions.get(symbol, Position(symbol)).quantity
        if side == OrderSide.BUY:
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity
        
        position_value = abs(new_position) * self.current_prices[symbol]
        max_position_value = self.account.equity * self.max_position_size
        
        if position_value > max_position_value:
            logger.error(f"Order would exceed maximum position size for {symbol}")
            return False
        
        return True
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get current account summary."""
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.account.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.account.positions.values())
        
        return {
            'cash': self.account.cash,
            'equity': self.account.equity,
            'buying_power': self.account.buying_power,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'num_positions': len(self.account.positions),
            'num_pending_orders': len(self.orders),
            'num_trades': len(self.trade_history)
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        
        positions = {}
        for symbol, position in self.account.positions.items():
            positions[symbol] = {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'current_price': self.current_prices.get(symbol, 0)
            }
        
        return positions
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get pending orders."""
        
        pending_orders = []
        for order in self.orders.values():
            pending_orders.append({
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'created_at': order.created_at.isoformat()
            })
        
        return pending_orders
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get trade history."""
        
        history = self.trade_history.copy()
        
        if limit:
            history = history[-limit:]
        
        # Convert datetime objects to strings
        for trade in history:
            if isinstance(trade['timestamp'], datetime):
                trade['timestamp'] = trade['timestamp'].isoformat()
        
        return history
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        if not self.trade_history:
            return {}
        
        # Calculate returns
        initial_equity = 100000.0  # Assuming initial capital
        current_equity = self.account.equity
        total_return = (current_equity - initial_equity) / initial_equity
        
        # Calculate trade-based metrics
        trade_pnls = []
        for trade in self.trade_history:
            # Simplified P&L calculation (would need more sophisticated tracking in reality)
            if trade['side'] == 'buy':
                trade_pnls.append(-trade['commission'])  # Cost of buying
            else:
                # This is simplified - real P&L would need position tracking
                trade_pnls.append(-trade['commission'])
        
        # Add unrealized P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.account.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.account.positions.values())
        
        win_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        total_trades = len(trade_pnls)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_commission': sum(trade['commission'] for trade in self.trade_history)
        }
    
    def reset_account(self, initial_cash: float = 100000.0) -> None:
        """Reset account to initial state."""
        
        self.account = Account(
            cash=initial_cash,
            equity=initial_cash,
            buying_power=initial_cash
        )
        
        self.orders.clear()
        self.order_history.clear()
        self.trade_history.clear()
        self.current_prices.clear()
        self.bid_ask_spreads.clear()
        
        logger.info(f"Account reset with ${initial_cash:,.2f} initial cash")