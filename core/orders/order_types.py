"""
Order Types & Execution - Features #90-100
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'
    OCO = 'oco'
    ICEBERG = 'iceberg'
    TWAP = 'twap'
    VWAP = 'vwap'

class MarketOrder:
    """Feature #90: Market Order"""
    def create(self, symbol: str, side: str, size: float) -> Dict:
        return {'type': 'MARKET', 'symbol': symbol, 'side': side, 'size': size, 'time': datetime.now().isoformat()}

class LimitOrder:
    """Feature #91: Limit Order"""
    def create(self, symbol: str, side: str, size: float, price: float) -> Dict:
        return {'type': 'LIMIT', 'symbol': symbol, 'side': side, 'size': size, 'price': price, 'time': datetime.now().isoformat()}

class StopOrder:
    """Feature #92: Stop Order"""
    def create(self, symbol: str, side: str, size: float, stop_price: float) -> Dict:
        return {'type': 'STOP', 'symbol': symbol, 'side': side, 'size': size, 'stop_price': stop_price}

class StopLimitOrder:
    """Feature #93: Stop-Limit Order"""
    def create(self, symbol: str, side: str, size: float, stop_price: float, limit_price: float) -> Dict:
        return {'type': 'STOP_LIMIT', 'symbol': symbol, 'side': side, 'size': size, 
                'stop_price': stop_price, 'limit_price': limit_price}

class TrailingStopOrder:
    """Feature #94: Trailing Stop Order"""
    def __init__(self):
        self.orders: Dict[str, Dict] = {}
    
    def create(self, order_id: str, symbol: str, side: str, size: float, trail_amount: float) -> Dict:
        order = {'type': 'TRAILING_STOP', 'symbol': symbol, 'side': side, 'size': size,
                 'trail_amount': trail_amount, 'peak_price': None}
        self.orders[order_id] = order
        return order
    
    def update_price(self, order_id: str, current_price: float) -> Optional[float]:
        if order_id not in self.orders:
            return None
        order = self.orders[order_id]
        if order['side'] == 'SELL':
            if order['peak_price'] is None or current_price > order['peak_price']:
                order['peak_price'] = current_price
            trigger = order['peak_price'] - order['trail_amount']
            if current_price <= trigger:
                return trigger
        else:  # BUY side trailing stop
            if order['peak_price'] is None or current_price < order['peak_price']:
                order['peak_price'] = current_price
            trigger = order['peak_price'] + order['trail_amount']
            if current_price >= trigger:
                return trigger
        return None

class OCOOrder:
    """Feature #95: One-Cancels-Other Order"""
    def create(self, symbol: str, side: str, size: float, take_profit: float, stop_loss: float) -> Dict:
        return {'type': 'OCO', 'symbol': symbol, 'side': side, 'size': size,
                'take_profit': take_profit, 'stop_loss': stop_loss, 'status': 'active'}

class IcebergOrder:
    """Feature #96: Iceberg Order"""
    def __init__(self):
        self.orders: Dict[str, Dict] = {}
    
    def create(self, order_id: str, symbol: str, side: str, total_size: float, visible_size: float, price: float) -> Dict:
        order = {'type': 'ICEBERG', 'symbol': symbol, 'side': side, 'total_size': total_size,
                 'visible_size': visible_size, 'filled': 0, 'price': price}
        self.orders[order_id] = order
        return order
    
    def get_next_slice(self, order_id: str) -> Optional[float]:
        if order_id not in self.orders:
            return None
        order = self.orders[order_id]
        remaining = order['total_size'] - order['filled']
        return min(order['visible_size'], remaining) if remaining > 0 else None
    
    def record_fill(self, order_id: str, filled: float):
        if order_id in self.orders:
            self.orders[order_id]['filled'] += filled

class BracketOrder:
    """Feature #97: Bracket Order"""
    def create(self, symbol: str, side: str, size: float, entry: float, stop_loss: float, take_profit: float) -> Dict:
        return {'type': 'BRACKET', 'symbol': symbol, 'side': side, 'size': size,
                'entry': entry, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'state': 'pending_entry'}

class ScaledOrder:
    """Feature #98: Scaled Order"""
    def create(self, symbol: str, side: str, total_size: float, levels: int, start_price: float, end_price: float) -> List[Dict]:
        orders = []
        size_per = total_size / levels
        price_step = (end_price - start_price) / (levels - 1) if levels > 1 else 0
        for i in range(levels):
            orders.append({'type': 'LIMIT', 'symbol': symbol, 'side': side,
                          'size': size_per, 'price': start_price + i * price_step})
        return orders

class TimeInForceManager:
    """Feature #99: Time-in-Force Manager"""
    TIF_TYPES = ['GTC', 'IOC', 'FOK', 'DAY', 'GTD']
    
    def validate(self, tif: str) -> bool:
        return tif in self.TIF_TYPES
    
    def is_expired(self, order: Dict) -> bool:
        tif = order.get('time_in_force', 'GTC')
        if tif == 'DAY':
            created = datetime.fromisoformat(order.get('created_at', datetime.now().isoformat()))
            return created.date() < datetime.now().date()
        if tif == 'GTD':
            expire_time = order.get('expire_time')
            if expire_time:
                return datetime.now() > datetime.fromisoformat(expire_time)
        return False

class OrderValidator:
    """Feature #100: Order Validator"""
    def __init__(self, min_size: float = 0.001, max_size: float = 1000):
        self.min_size = min_size
        self.max_size = max_size
    
    def validate(self, order: Dict) -> Dict:
        errors = []
        if order.get('size', 0) < self.min_size:
            errors.append('Size below minimum')
        if order.get('size', 0) > self.max_size:
            errors.append('Size above maximum')
        if order.get('price', 0) <= 0 and order.get('type') == 'LIMIT':
            errors.append('Invalid price')
        return {'valid': len(errors) == 0, 'errors': errors}

# Factories
def get_market_order(): return MarketOrder()
def get_limit_order(): return LimitOrder()
def get_stop_order(): return StopOrder()
def get_stop_limit_order(): return StopLimitOrder()
def get_trailing_stop(): return TrailingStopOrder()
def get_oco_order(): return OCOOrder()
def get_iceberg_order(): return IcebergOrder()
def get_bracket_order(): return BracketOrder()
def get_scaled_order(): return ScaledOrder()
def get_tif_manager(): return TimeInForceManager()
def get_order_validator(): return OrderValidator()
