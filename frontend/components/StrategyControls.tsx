'use client';

import { useState, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
    const [scalperMode, setScalperMode] = useState(false);
    const [smcMode, setSmcMode] = useState<'balanced' | 'aggressive' | 'conservative'>('balanced');

    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const response = await fetch(`${API_URL}/api/strategies`);
                const result = await response.json();
                // Handle wrapped response format
                const data = result.data?.strategies || result.strategies || [];
                setStrategies(data);
                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch strategies:', error);
                // NO MOCK DATA - show empty for new accounts
                setStrategies([]);
                setLoading(false);
            }
        };

        fetchStrategies();
    }, []);

    const toggleStrategy = async (strategyName: string) => {
        try {
            const strategy = strategies.find((s) => s.name === strategyName);
            const endpoint = strategy?.enabled ? `${API_URL}/api/strategy/disable` : `${API_URL}/api/strategy/enable`;

            await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategy: strategyName }),
            });

            setStrategies((prev) =>
                prev.map((s) =>
                    s.name === strategyName ? { ...s, enabled: !s.enabled } : s
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

            <div className="mb-4 border border-border rounded p-3 space-y-3">
                <div className="flex items-center justify-between">
                    <div>
                        <div className="font-medium text-sm">Scalper Mode</div>
                        <div className="text-text-secondary text-xs">Enable fast intraday execution profile</div>
                    </div>
                    <button
                        onClick={() => setScalperMode((value) => !value)}
                        className={`px-3 py-1 rounded text-sm font-medium transition-colors ${scalperMode
                            ? 'bg-accent-green text-white hover:bg-accent-green/80'
                            : 'bg-bg-tertiary text-text-secondary hover:bg-bg-tertiary/80'
                            }`}
                    >
                        {scalperMode ? 'ON' : 'OFF'}
                    </button>
                </div>

                <div className="flex items-center justify-between gap-3">
                    <div>
                        <div className="font-medium text-sm">SMC Mode Selector</div>
                        <div className="text-text-secondary text-xs">Adjust confluence strictness for SMC entries</div>
                    </div>
                    <select
                        value={smcMode}
                        onChange={(event) => setSmcMode(event.target.value as 'balanced' | 'aggressive' | 'conservative')}
                        className="bg-bg-tertiary border border-border rounded px-2 py-1 text-sm"
                    >
                        <option value="conservative">Conservative</option>
                        <option value="balanced">Balanced</option>
                        <option value="aggressive">Aggressive</option>
                    </select>
                </div>
            </div>

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
