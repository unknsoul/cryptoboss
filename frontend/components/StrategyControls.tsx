'use client';

import { useState, useEffect } from 'react';

interface Strategy {
    name: string;
    enabled: boolean;
    performance: {
        win_rate: number;
        pnl: number;
        trades: number;
    };
}

export default function StrategyControls() {
    const [strategies, setStrategies] = useState<Strategy[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const response = await fetch('/api/strategies');
                const data = await response.json();
                setStrategies(data);
                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch strategies:', error);
                // Mock data for demo
                setStrategies([
                    {
                        name: 'Momentum Strategy',
                        enabled: true,
                        performance: { win_rate: 65.0, pnl: 450.25, trades: 82 },
                    },
                    {
                        name: 'Mean Reversion',
                        enabled: true,
                        performance: { win_rate: 58.5, pnl: 320.50, trades: 105 },
                    },
                    {
                        name: 'Breakout Strategy',
                        enabled: false,
                        performance: { win_rate: 52.0, pnl: -85.75, trades: 45 },
                    },
                ]);
                setLoading(false);
            }
        };

        fetchStrategies();
    }, []);

    const toggleStrategy = async (strategyName: string) => {
        try {
            const strategy = strategies.find((s) => s.name === strategyName);
            const endpoint = strategy?.enabled ? '/api/strategy/disable' : '/api/strategy/enable';

            await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategy: strategyName }),
            });

            setStrategies((prev) =>
                prev.map((s) =>
                    s.name === strategy Name ? { ...s, enabled: !s.enabled } : s
                )
            );
        } catch (error) {
            console.error('Failed to toggle strategy:', error);
        }
    };

    if (loading) {
        return (
            <div className="card">
                <h2 className="text-xl font-bold mb-4">Strategy Controls</h2>
                <div className="text-center py-8 text-text-secondary">Loading strategies...</div>
            </div>
        );
    }

    return (
        <div className="card">
            <h2 className="text-xl font-bold mb-4">Strategy Controls</h2>

            <div className="space-y-4">
                {strategies.map((strategy) => (
                    <div key={strategy.name} className="border border-border rounded p-3">
                        <div className="flex justify-between items-start mb-3">
                            <div>
                                <div className="font-medium">{strategy.name}</div>
                                <div className="text-text-secondary text-xs mt-1">
                                    {strategy.performance.trades} trades
                                </div>
                            </div>

                            <button
                                onClick={() => toggleStrategy(strategy.name)}
                                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${strategy.enabled
                                        ? 'bg-accent-green text-white hover:bg-accent-green/80'
                                        : 'bg-bg-tertiary text-text-secondary hover:bg-bg-tertiary/80'
                                    }`}
                            >
                                {strategy.enabled ? 'Enabled' : 'Disabled'}
                            </button>
                        </div>

                        {/* Performance Stats */}
                        <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>
                                <div className="text-text-secondary text-xs">Win Rate</div>
                                <div className="font-mono">{strategy.performance.win_rate.toFixed(1)}%</div>
                            </div>
                            <div>
                                <div className="text-text-secondary text-xs">P&L</div>
                                <div
                                    className={`font-mono ${strategy.performance.pnl >= 0 ? 'status-long' : 'status-short'
                                        }`}
                                >
                                    {strategy.performance.pnl >= 0 ? '+' : ''}${strategy.performance.pnl.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Add Strategy Button */}
            <button className="btn btn-secondary w-full mt-4">
                + Add Strategy
            </button>
        </div>
    );
}
