'use client';

import { useState, useEffect } from 'react';

interface Position {
    id: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    entry_price: number;
    current_price: number;
    size: number;
    pnl: number;
    pnl_pct: number;
    unrealized_pnl: number;
}

export default function PositionsTable() {
    const [positions, setPositions] = useState<Position[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchPositions = async () => {
            try {
                const response = await fetch('/api/positions');
                const data = await response.json();
                setPositions(data);
                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch positions:', error);
                // Mock data for demo
                setPositions([
                    {
                        id: '1',
                        symbol: 'BTCUSDT',
                        side: 'LONG',
                        entry_price: 41250.50,
                        current_price: 41450.25,
                        size: 0.5,
                        pnl: 99.88,
                        pnl_pct: 0.48,
                        unrealized_pnl: 99.88,
                    },
                ]);
                setLoading(false);
            }
        };

        fetchPositions();
        const interval = setInterval(fetchPositions, 5000); // Update every 5s
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="card">
                <h2 className="text-xl font-bold mb-4">Open Positions</h2>
                <div className="text-center py-8 text-text-secondary">Loading positions...</div>
            </div>
        );
    }

    if (positions.length === 0) {
        return (
            <div className="card">
                <h2 className="text-xl font-bold mb-4">Open Positions</h2>
                <div className="text-center py-8 text-text-secondary">
                    No open positions
                </div>
            </div>
        );
    }

    const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0);

    return (
        <div className="card">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">Open Positions</h2>
                <div className="flex items-center gap-2">
                    <span className="text-text-secondary text-sm">Total P&L:</span>
                    <span className={`font-mono text-lg ${totalPnL >= 0 ? 'status-long' : 'status-short'}`}>
                        {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
                    </span>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry Price</th>
                            <th>Current Price</th>
                            <th className="text-right">Size</th>
                            <th className="text-right">P&L</th>
                            <th className="text-right">P&L %</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions.map((position) => (
                            <tr key={position.id}>
                                <td className="font-medium">{position.symbol}</td>
                                <td>
                                    <span
                                        className={`px-2 py-1 rounded text-xs font-medium ${position.side === 'LONG'
                                                ? 'bg-accent-green/20 text-accent-green'
                                                : 'bg-accent-red/20 text-accent-red'
                                            }`}
                                    >
                                        {position.side}
                                    </span>
                                </td>
                                <td className="font-mono">${position.entry_price.toFixed(2)}</td>
                                <td className="font-mono">${position.current_price.toFixed(2)}</td>
                                <td className="text-right font-mono">{position.size.toFixed(4)}</td>
                                <td className={`text-right font-mono ${position.pnl >= 0 ? 'status-long' : 'status-short'}`}>
                                    {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                                </td>
                                <td className={`text-right font-mono ${position.pnl_pct >= 0 ? 'status-long' : 'status-short'}`}>
                                    {position.pnl_pct >= 0 ? '+' : ''}{position.pnl_pct.toFixed(2)}%
                                </td>
                                <td>
                                    <button className="btn btn-danger text-xs">Close</button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
