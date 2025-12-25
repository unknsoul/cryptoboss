"""
User Interface Components - Enterprise Features #350, #355, #358, #360
Dashboard Widgets, Charts, Trade History, System Status.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class DashboardWidgets:
    """
    Feature #350: Dashboard Widgets
    
    Modular widget components for the trading dashboard.
    """
    
    def __init__(self):
        """Initialize dashboard widgets."""
        self.widgets: Dict[str, Dict] = {}
        self.widget_data: Dict[str, Any] = {}
        self.refresh_intervals: Dict[str, int] = {}
        
        logger.info("Dashboard Widgets initialized")
    
    def register_widget(
        self,
        widget_id: str,
        widget_type: str,
        title: str,
        refresh_seconds: int = 5,
        config: Optional[Dict] = None
    ):
        """Register a dashboard widget."""
        self.widgets[widget_id] = {
            'id': widget_id,
            'type': widget_type,
            'title': title,
            'config': config or {},
            'created_at': datetime.now().isoformat()
        }
        self.refresh_intervals[widget_id] = refresh_seconds
    
    def update_widget_data(self, widget_id: str, data: Any):
        """Update widget data."""
        self.widget_data[widget_id] = {
            'data': data,
            'updated_at': datetime.now().isoformat()
        }
    
    def get_widget(self, widget_id: str) -> Optional[Dict]:
        """Get widget with current data."""
        if widget_id not in self.widgets:
            return None
        
        return {
            **self.widgets[widget_id],
            **self.widget_data.get(widget_id, {})
        }
    
    def get_all_widgets(self) -> List[Dict]:
        """Get all widgets with data."""
        return [self.get_widget(wid) for wid in self.widgets]
    
    def create_equity_widget(self, equity: float, daily_pnl: float) -> Dict:
        """Create equity summary widget."""
        widget_id = 'equity_summary'
        
        self.register_widget(widget_id, 'stat_card', 'Account Equity')
        self.update_widget_data(widget_id, {
            'value': f"${equity:,.2f}",
            'change': f"{'+' if daily_pnl > 0 else ''}{daily_pnl:,.2f}",
            'change_type': 'positive' if daily_pnl >= 0 else 'negative'
        })
        
        return self.get_widget(widget_id)
    
    def create_position_widget(self, positions: List[Dict]) -> Dict:
        """Create active positions widget."""
        widget_id = 'active_positions'
        
        self.register_widget(widget_id, 'table', 'Active Positions')
        self.update_widget_data(widget_id, {
            'columns': ['Symbol', 'Side', 'Size', 'Entry', 'P&L'],
            'rows': [
                [p['symbol'], p['side'], p['size'], f"${p['entry']:,.2f}", f"${p.get('pnl', 0):,.2f}"]
                for p in positions
            ]
        })
        
        return self.get_widget(widget_id)
    
    def create_stats_widget(self, metrics: Dict) -> Dict:
        """Create trading stats widget."""
        widget_id = 'trading_stats'
        
        self.register_widget(widget_id, 'stats_grid', 'Trading Statistics')
        self.update_widget_data(widget_id, {
            'items': [
                {'label': 'Total Trades', 'value': metrics.get('total_trades', 0)},
                {'label': 'Win Rate', 'value': f"{metrics.get('win_rate', 0):.1%}"},
                {'label': 'Profit Factor', 'value': f"{metrics.get('profit_factor', 0):.2f}"},
                {'label': 'Sharpe Ratio', 'value': f"{metrics.get('sharpe', 0):.2f}"}
            ]
        })
        
        return self.get_widget(widget_id)


class RealTimeChartsData:
    """
    Feature #355: Real-Time Charts Data
    
    Provides real-time data for charting components.
    """
    
    def __init__(self, max_points: int = 500):
        """
        Initialize chart data provider.
        
        Args:
            max_points: Maximum data points to retain
        """
        self.max_points = max_points
        self.price_data: Dict[str, List[Dict]] = defaultdict(list)
        self.indicator_data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self.trade_markers: Dict[str, List[Dict]] = defaultdict(list)
        
        logger.info("Real-Time Charts Data initialized")
    
    def add_price_point(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0
    ):
        """Add OHLCV data point."""
        self.price_data[symbol].append({
            'time': timestamp.isoformat(),
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # Trim to max points
        self.price_data[symbol] = self.price_data[symbol][-self.max_points:]
    
    def add_indicator(
        self,
        symbol: str,
        indicator_name: str,
        timestamp: datetime,
        value: float
    ):
        """Add indicator data point."""
        self.indicator_data[symbol][indicator_name].append({
            'time': timestamp.isoformat(),
            'value': value
        })
        
        self.indicator_data[symbol][indicator_name] = \
            self.indicator_data[symbol][indicator_name][-self.max_points:]
    
    def add_trade_marker(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        side: str,
        label: str = ''
    ):
        """Add trade marker for chart."""
        self.trade_markers[symbol].append({
            'time': timestamp.isoformat(),
            'price': price,
            'side': side,
            'label': label
        })
        
        self.trade_markers[symbol] = self.trade_markers[symbol][-100:]
    
    def get_chart_data(self, symbol: str) -> Dict:
        """Get complete chart data for a symbol."""
        return {
            'symbol': symbol,
            'candles': self.price_data.get(symbol, []),
            'indicators': dict(self.indicator_data.get(symbol, {})),
            'trades': self.trade_markers.get(symbol, [])
        }
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        data = self.price_data.get(symbol, [])
        return data[-1]['close'] if data else None


class TradeHistoryView:
    """
    Feature #358: Trade History View
    
    Manages trade history display and filtering.
    """
    
    def __init__(self):
        """Initialize trade history view."""
        self.trades: List[Dict] = []
        self.filters: Dict[str, Any] = {}
        
        logger.info("Trade History View initialized")
    
    def add_trade(self, trade: Dict):
        """Add a trade to history."""
        entry = {
            **trade,
            'id': len(self.trades) + 1,
            'timestamp': trade.get('timestamp', datetime.now().isoformat())
        }
        self.trades.append(entry)
    
    def set_filter(self, **filters):
        """Set display filters."""
        self.filters = filters
    
    def get_filtered_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        side: Optional[str] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get filtered trade history."""
        filtered = self.trades
        
        if side:
            filtered = [t for t in filtered if t.get('side') == side]
        
        if min_pnl is not None:
            filtered = [t for t in filtered if t.get('pnl', 0) >= min_pnl]
        
        if max_pnl is not None:
            filtered = [t for t in filtered if t.get('pnl', 0) <= max_pnl]
        
        if start_date:
            start_str = start_date.isoformat()
            filtered = [t for t in filtered if t.get('timestamp', '') >= start_str]
        
        if end_date:
            end_str = end_date.isoformat()
            filtered = [t for t in filtered if t.get('timestamp', '') <= end_str]
        
        # Sort by timestamp descending
        filtered = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return filtered[offset:offset + limit]
    
    def get_summary(self) -> Dict:
        """Get trade history summary."""
        if not self.trades:
            return {'total': 0}
        
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        return {
            'total': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / len(self.trades) * 100, 1),
            'total_pnl': round(sum(t.get('pnl', 0) for t in self.trades), 2),
            'avg_win': round(sum(t['pnl'] for t in wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0
        }
    
    def export_csv(self) -> str:
        """Export trade history as CSV string."""
        if not self.trades:
            return ""
        
        headers = ['ID', 'Timestamp', 'Side', 'Entry', 'Exit', 'Size', 'P&L']
        lines = [','.join(headers)]
        
        for t in self.trades:
            row = [
                str(t.get('id', '')),
                t.get('timestamp', ''),
                t.get('side', ''),
                str(t.get('entry_price', '')),
                str(t.get('exit_price', '')),
                str(t.get('size', '')),
                str(t.get('pnl', ''))
            ]
            lines.append(','.join(row))
        
        return '\n'.join(lines)


class SystemStatusPanel:
    """
    Feature #360: System Status Panel
    
    Displays overall system health and status.
    """
    
    def __init__(self):
        """Initialize system status panel."""
        self.components: Dict[str, Dict] = {}
        self.alerts: List[Dict] = []
        self.last_update: Optional[datetime] = None
        
        logger.info("System Status Panel initialized")
    
    def register_component(self, name: str, component_type: str):
        """Register a system component."""
        self.components[name] = {
            'name': name,
            'type': component_type,
            'status': 'unknown',
            'last_check': None,
            'details': {}
        }
    
    def update_status(self, name: str, status: str, details: Optional[Dict] = None):
        """Update component status."""
        if name in self.components:
            self.components[name]['status'] = status
            self.components[name]['last_check'] = datetime.now().isoformat()
            if details:
                self.components[name]['details'] = details
        
        self.last_update = datetime.now()
    
    def add_alert(self, severity: str, message: str, component: str = ''):
        """Add a system alert."""
        self.alerts.append({
            'severity': severity,
            'message': message,
            'component': component,
            'timestamp': datetime.now().isoformat()
        })
        self.alerts = self.alerts[-50:]  # Keep last 50
    
    def get_overall_status(self) -> str:
        """Get overall system status."""
        if not self.components:
            return 'unknown'
        
        statuses = [c['status'] for c in self.components.values()]
        
        if 'critical' in statuses:
            return 'critical'
        if 'error' in statuses:
            return 'degraded'
        if 'warning' in statuses:
            return 'warning'
        if all(s == 'healthy' for s in statuses):
            return 'healthy'
        
        return 'unknown'
    
    def get_panel_data(self) -> Dict:
        """Get complete panel data."""
        return {
            'overall_status': self.get_overall_status(),
            'components': list(self.components.values()),
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def get_component_health(self, name: str) -> Dict:
        """Get specific component health."""
        return self.components.get(name, {'status': 'unknown'})


# Singletons
_widgets: Optional[DashboardWidgets] = None
_charts: Optional[RealTimeChartsData] = None
_history: Optional[TradeHistoryView] = None
_status: Optional[SystemStatusPanel] = None


def get_dashboard_widgets() -> DashboardWidgets:
    global _widgets
    if _widgets is None:
        _widgets = DashboardWidgets()
    return _widgets


def get_charts_data() -> RealTimeChartsData:
    global _charts
    if _charts is None:
        _charts = RealTimeChartsData()
    return _charts


def get_trade_history() -> TradeHistoryView:
    global _history
    if _history is None:
        _history = TradeHistoryView()
    return _history


def get_system_status() -> SystemStatusPanel:
    global _status
    if _status is None:
        _status = SystemStatusPanel()
    return _status


if __name__ == '__main__':
    # Test widgets
    widgets = DashboardWidgets()
    equity_widget = widgets.create_equity_widget(10500, 150)
    print(f"Equity widget: {equity_widget}")
    
    # Test charts
    charts = RealTimeChartsData()
    charts.add_price_point('BTCUSDT', datetime.now(), 50000, 50100, 49900, 50050, 100)
    charts.add_indicator('BTCUSDT', 'RSI', datetime.now(), 55.5)
    print(f"Chart data: {charts.get_chart_data('BTCUSDT')}")
    
    # Test history
    history = TradeHistoryView()
    history.add_trade({'side': 'LONG', 'pnl': 50, 'entry_price': 50000, 'exit_price': 50100})
    print(f"Trade summary: {history.get_summary()}")
    
    # Test status
    status = SystemStatusPanel()
    status.register_component('exchange', 'api')
    status.update_status('exchange', 'healthy', {'latency_ms': 50})
    print(f"System status: {status.get_panel_data()}")
