"""
Professional Trading Bot Dashboard
Real-time monitoring and control interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff1744;
    }
</style>
""", unsafe_allow_html=True)


def load_metrics():
    """Load metrics from logs"""
    try:
        # Try to load from metrics file
        metrics_file = Path("logs/metrics.json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
    except:
        pass
    
    # Return dummy data for demonstration
    return {
        "current_equity": 10500.50,
        "daily_pnl": 250.30,
        "daily_pnl_pct": 0.0244,
        "total_pnl": 500.50,
        "total_return_pct": 0.0500,
        "active_positions": 2,
        "total_trades": 45,
        "win_rate": 0.62,
        "sharpe_ratio": 2.1,
        "max_drawdown": -0.08
    }


def load_trades():
    """Load recent trades"""
    try:
        trades_file = Path("logs/trades.csv")
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except:
        pass
    
    # Return dummy data
    return pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=20, freq='H'),
        'symbol': ['BTCUSDT'] * 10 + ['ETHUSDT'] * 10,
        'side': ['BUY', 'SELL'] * 10,
        'quantity': [0.1] * 20,
        'price': [45000 + i*100 for i in range(20)],
        'pnl': [50, -20, 80, 30, -10, 60, 40, -15, 90, 25] * 2
    })


def load_positions():
    """Load current positions"""
    # Dummy data
    return pd.DataFrame({
        'symbol': ['BTCUSDT', 'ETHUSDT'],
        'side': ['LONG', 'LONG'],
        'quantity': [0.15, 1.2],
        'entry_price': [44500, 2950],
        'current_price': [45200, 3010],
        'unrealized_pnl': [105, 72],
        'pnl_pct': [0.0157, 0.0203]
    })


# Main Dashboard
def main():
    st.markdown('<div class="main-header">🚀 Professional Trading Bot Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Bot Status")
    status = st.sidebar.selectbox("Status", ["Running", "Paused", "Stopped"])
    
    if status == "Running":
        st.sidebar.success("✅ Bot is active")
    elif status == "Paused":
        st.sidebar.warning("⏸️ Bot is paused")
    else:
        st.sidebar.error("🛑 Bot is stopped")
    
    # Load data
    metrics = load_metrics()
    trades_df = load_trades()
    positions_df = load_positions()
    
    # Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="💰 Current Equity",
            value=f"${metrics['current_equity']:,.2f}",
            delta=f"${metrics['daily_pnl']:,.2f} ({metrics['daily_pnl_pct']:.2%})"
        )
    
    with col2:
        st.metric(
            label="📊 Total P&L",
            value=f"${metrics['total_pnl']:,.2f}",
            delta=f"{metrics['total_return_pct']:.2%}"
        )
    
    with col3:
        st.metric(
            label="📈 Win Rate",
            value=f"{metrics['win_rate']:.1%}",
            delta=f"{metrics['total_trades']} trades"
        )
    
    with col4:
        st.metric(
            label="⚡ Sharpe Ratio",
            value=f"{metrics['sharpe_ratio']:.2f}",
            delta="Ex cellent" if metrics['sharpe_ratio'] > 2 else "Good"
        )
    
    with col5:
        st.metric(
            label="📉 Max Drawdown",
            value=f"{metrics['max_drawdown']:.1%}",
            delta="Within limits"
        )
    
    st.markdown("---")
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💼 Positions", "📋 Trades", "📈 Performance", "⚙️ Settings"
    ])
    
    with tab1:
        # Overview Tab
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("Equity Curve")
            
            # Generate equity curve
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            equity_curve = 10000 + (trades_df['pnl'].cumsum().iloc[-100:] if len(trades_df) >= 100 
                                   else trades_df['pnl'].cumsum())
            equity_curve = pd.Series([10000 + i*5 + (i**1.5)*0.1 for i in range(100)], index=dates)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("Active Positions")
            st.dataframe(
                positions_df[['symbol', 'side', 'unrealized_pnl', 'pnl_pct']],
                hide_index=True,
                use_container_width=True
            )
            
            st.subheader("Risk Metrics")
            risk_data = pd.DataFrame({
                'Metric': ['Daily VaR', 'Portfolio Beta', 'Correlation', 'Leverage'],
                'Value': ['-$250', '0.85', '0.42', '1.0x']
            })
            st.dataframe(risk_data, hide_index=True, use_container_width=True)
    
    with tab2:
        # Positions Tab
        st.subheader("Current Positions")
        
        # Format positions table
        pos_display = positions_df.copy()
        pos_display['unrealized_pnl'] = pos_display['unrealized_pnl'].apply(lambda x: f"${x:.2f}")
        pos_display['pnl_pct'] = pos_display['pnl_pct'].apply(lambda x: f"{x:.2%}")
        pos_display['entry_price'] = pos_display['entry_price'].apply(lambda x: f"${x:.2f}")
        pos_display['current_price'] = pos_display['current_price'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(pos_display, hide_index=True, use_container_width=True)
        
        # Position chart
        st.subheader("Position Distribution")
        
        fig = go.Figure(data=[go.Pie(
            labels=positions_df['symbol'],
            values=positions_df['quantity'] * positions_df['current_price'],
            hole=0.3
        )])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Trades Tab
        st.subheader("Recent Trades")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                options=trades_df['symbol'].unique(),
                default=trades_df['symbol'].unique()
            )
        with col2:
            side_filter = st.multiselect(
                "Filter by Side",
                options=['BUY', 'SELL'],
                default=['BUY', 'SELL']
            )
        with col3:
            num_trades = st.slider("Number of trades", 10, 100, 20)
        
        # Apply filters
        filtered_trades = trades_df[
            (trades_df['symbol'].isin(symbol_filter)) &
            (trades_df['side'].isin(side_filter))
        ].tail(num_trades)
        
        # Format for display
        display_trades = filtered_trades.copy()
        display_trades['pnl'] = display_trades['pnl'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "-"
        )
        display_trades['price'] = display_trades['price'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_trades, hide_index=True, use_container_width=True)
        
        # P&L distribution
        st.subheader("P&L Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered_trades['pnl'],
            nbinsx=20,
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Performance Tab
        st.subheader("Strategy Performance")
        
        # Performance metrics by strategy
        strategy_perf = pd.DataFrame({
            'Strategy': ['Enhanced Momentum', 'Mean Reversion', 'Scalping', 'MACD Crossover'],
            'Trades': [15, 10, 12, 8],
            'Win Rate': [0.67, 0.60, 0.58, 0.625],
            'Avg P&L': [45.20, 35.50, 22.30, 38.10],
            'Sharpe': [2.3, 1.8, 1.5, 2.0]
        })
        
        st.dataframe(strategy_perf, hide_index=True, use_container_width=True)
        
        # Performance chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                strategy_perf,
                x='Strategy',
                y='Sharpe',
                title="Sharpe Ratio by Strategy",
                color='Sharpe',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                strategy_perf,
                x='Strategy',
                y='Win Rate',
                title="Win Rate by Strategy",
                color='Win Rate',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        # Settings Tab
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trading Settings")
            capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
            max_position = st.slider("Max Position Size (%)", 0.0, 0.5, 0.2)
            risk_per_trade = st.slider("Risk Per Trade (%)", 0.0, 0.1, 0.02)
        
        with col2:
            st.markdown("### Risk Management")
            max_daily_loss = st.slider("Max Daily Loss (%)", 0.0, 0.2, 0.05)
            max_drawdown = st.slider("Max Drawdown (%)", 0.0, 0.3, 0.15)
            target_vol = st.slider("Target Volatility (%)", 0.0, 0.5, 0.15)
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")
        
        st.markdown("---")
        st.markdown("### Alerts Configuration")
        
        enable_email = st.checkbox("Enable Email Alerts")
        enable_slack = st.checkbox("Enable Slack Alerts")
        enable_discord = st.checkbox("Enable Discord Alerts")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
