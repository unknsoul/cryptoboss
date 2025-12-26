'use client';

import { useState, useEffect } from 'react';

interface RiskMetric {
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
    daily_pnl: number;
}

export default function RiskMetrics() {
    const [metrics, setMetrics] = useState<RiskMetric | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await fetch('/api/risk-metrics');
                const data = await response.json();
                setMetrics(data);
                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch risk metrics:', error);
                // Mock data for demo
                setMetrics({
                    sharpe_ratio: 1.85,
                    max_drawdown: -8.5,
                    win_rate: 62.5,
                    profit_factor: 1.78,
                    total_trades: 245,
                    daily_pnl: 125.50,
                });
                setLoading(false);
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 10000); // Update every 10s
        return () => clearInterval(interval);
    }, []);

    if (loading || !metrics) {
        return (
            <div className="card">
                <h2 className="text-xl font-bold mb-4">Risk Metrics</h2>
                <div className="text-center py-8 text-text-secondary">Loading metrics...</div>
            </div>
        );
    }

    const metricItems = [
        {
            label: 'Sharpe Ratio',
            value: metrics.sharpe_ratio.toFixed(2),
            good: metrics.sharpe_ratio > 1.0,
            suffix: '',
        },
        {
            label: 'Max Drawdown',
            value: metrics.max_drawdown.toFixed(2),
            good: metrics.max_drawdown > -10,
            suffix: '%',
        },
        {
            label: 'Win Rate',
            value: metrics.win_rate.toFixed(1),
            good: metrics.win_rate > 50,
            suffix: '%',
        },
        {
            label: 'Profit Factor',
            value: metrics.profit_factor.toFixed(2),
            good: metrics.profit_factor > 1.5,
            suffix: '',
        },
        {
            label: 'Total Trades',
            value: metrics.total_trades.toString(),
            good: true,
            suffix: '',
        },
        {
            label: 'Daily P&L',
            value: metrics.daily_pnl >= 0 ? `+${metrics.daily_pnl.toFixed(2)}` : metrics.daily_pnl.toFixed(2),
            good: metrics.daily_pnl >= 0,
            suffix: '',
            prefix: '$',
        },
    ];

    return (
        <div className="card">
            <h2 className="text-xl font-bold mb-4">Risk Metrics</h2>

            <div className="space-y-4">
                {metricItems.map((item) => (
                    <div key={item.label} className="border-b border-border pb-3 last:border-0">
                        <div className="text-text-secondary text-sm mb-1">{item.label}</div>
                        <div className={`text-2xl font-mono font-bold ${item.good ? 'status-long' : 'status-short'}`}>
                            {item.prefix || ''}
                            {item.value}
                            {item.suffix}
                        </div>
                    </div>
                ))}
            </div>

            {/* Risk Warning */}
            {metrics.max_drawdown < -10 && (
                <div className="mt-4 p-3 bg-accent-red/10 border border-accent-red rounded">
                    <div className="text-accent-red text-sm font-medium">⚠️ High Drawdown Warning</div>
                    <div className="text-text-secondary text-xs mt-1">
                        Consider reducing position sizes or pausing trading
                    </div>
                </div>
            )}
        </div>
    );
}
