'use client';

/**
 * Positions Page
 * 
 * Purpose: Trade transparency
 * Rules:
 * - No manual close buttons in live mode
 * - Clear separation between paper and live
 * - Entry reasoning visible
 */

// Mock positions data
const positions = [
    {
        id: 1,
        symbol: 'BTC/USDT',
        side: 'LONG',
        entryPrice: 88920.00,
        currentPrice: 89168.42,
        size: 0.025,
        exposure: 2229.21,
        unrealizedPnL: 6.21,
        pnlPercent: 0.28,
        entryTime: '2024-01-28 12:45:00',
        entryReason: 'Bias: LONG_BIAS (72%), Context: TRENDING_UP, Signal: Higher low confirmed',
        stopLoss: 87500.00,
        takeProfit: 92000.00,
        mode: 'paper' as const,
    },
    {
        id: 2,
        symbol: 'ETH/USDT',
        side: 'LONG',
        entryPrice: 3150.00,
        currentPrice: 3185.50,
        size: 0.7,
        exposure: 2229.85,
        unrealizedPnL: 24.85,
        pnlPercent: 1.13,
        entryTime: '2024-01-28 11:30:00',
        entryReason: 'Bias: LONG_BIAS (68%), Context: RANGING, Signal: Support bounce',
        stopLoss: 3050.00,
        takeProfit: 3350.00,
        mode: 'paper' as const,
    },
];

const closedPositions = [
    {
        id: 101,
        symbol: 'BTC/USDT',
        side: 'SHORT',
        entryPrice: 89500.00,
        exitPrice: 89050.00,
        size: 0.02,
        realizedPnL: 9.00,
        pnlPercent: 0.50,
        entryTime: '2024-01-28 09:00:00',
        exitTime: '2024-01-28 10:30:00',
        exitReason: 'Take profit reached',
        mode: 'paper' as const,
    },
];

function PositionCard({ position, isOpen = true }: { position: typeof positions[0]; isOpen?: boolean }) {
    const pnlColor = position.unrealizedPnL >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]';
    const sideColor = position.side === 'LONG' ? 'badge-success' : 'badge-danger';

    return (
        <div className="card">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <span className="text-[#e7e9ea] font-medium">{position.symbol}</span>
                    <span className={`badge ${sideColor}`}>{position.side}</span>
                    <span className="badge badge-neutral text-xs">{position.mode.toUpperCase()}</span>
                </div>
                <span className={`value-md ${pnlColor}`}>
                    {position.unrealizedPnL >= 0 ? '+' : ''}${position.unrealizedPnL.toFixed(2)}
                    <span className="text-sm ml-1">({position.pnlPercent.toFixed(2)}%)</span>
                </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div>
                    <span className="label block">Entry Price</span>
                    <span className="text-[#e7e9ea]">${position.entryPrice.toLocaleString()}</span>
                </div>
                <div>
                    <span className="label block">Current Price</span>
                    <span className="text-[#e7e9ea]">${position.currentPrice.toLocaleString()}</span>
                </div>
                <div>
                    <span className="label block">Size</span>
                    <span className="text-[#e7e9ea]">{position.size}</span>
                </div>
                <div>
                    <span className="label block">Exposure</span>
                    <span className="text-[#e7e9ea]">${position.exposure.toLocaleString()}</span>
                </div>
            </div>

            {/* Entry Reasoning - Key for explainability */}
            <div className="bg-[#1a1f26] rounded-md p-3 mb-4">
                <span className="label block mb-1">Entry Reasoning</span>
                <p className="text-sm text-[#8b98a5]">{position.entryReason}</p>
            </div>

            {/* Exit Conditions */}
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <span className="label block">Stop Loss</span>
                    <span className="text-[#a65454]">${position.stopLoss.toLocaleString()}</span>
                </div>
                <div>
                    <span className="label block">Take Profit</span>
                    <span className="text-[#4a9268]">${position.takeProfit.toLocaleString()}</span>
                </div>
            </div>

            <div className="mt-4 pt-4 border-t border-[#2d3640]">
                <span className="text-xs text-[#6b7280]">Opened: {position.entryTime}</span>
            </div>
        </div>
    );
}

export default function PositionsPage() {
    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Positions</h1>
                <p className="text-[#8b98a5] text-sm">
                    Trade transparency — every position is explainable
                </p>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="card text-center">
                    <span className="label">Open Positions</span>
                    <span className="value-lg block mt-1">{positions.length}</span>
                </div>
                <div className="card text-center">
                    <span className="label">Total Exposure</span>
                    <span className="value-lg block mt-1">
                        ${positions.reduce((sum, p) => sum + p.exposure, 0).toLocaleString()}
                    </span>
                </div>
                <div className="card text-center">
                    <span className="label">Unrealized P&L</span>
                    <span className={`value-lg block mt-1 ${positions.reduce((sum, p) => sum + p.unrealizedPnL, 0) >= 0
                            ? 'text-[#4a9268]' : 'text-[#a65454]'
                        }`}>
                        +${positions.reduce((sum, p) => sum + p.unrealizedPnL, 0).toFixed(2)}
                    </span>
                </div>
                <div className="card text-center">
                    <span className="label">Today's Closed</span>
                    <span className="value-lg block mt-1">{closedPositions.length}</span>
                </div>
            </div>

            {/* Open Positions */}
            <div>
                <h2 className="heading-md mb-4">Open Positions</h2>
                {positions.length > 0 ? (
                    <div className="space-y-4">
                        {positions.map((position) => (
                            <PositionCard key={position.id} position={position} isOpen={true} />
                        ))}
                    </div>
                ) : (
                    <div className="empty-state">
                        <div className="empty-state-icon">📭</div>
                        <div className="empty-state-title">No Open Positions</div>
                        <div className="empty-state-description">
                            The system has no active positions. This is a normal state.
                        </div>
                    </div>
                )}
            </div>

            {/* Closed Positions Today */}
            <div className="mt-8">
                <h2 className="heading-md mb-4">Closed Today</h2>
                <div className="card">
                    <table className="table w-full">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>P&L</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody>
                            {closedPositions.map((pos) => (
                                <tr key={pos.id}>
                                    <td>{pos.symbol}</td>
                                    <td>
                                        <span className={`badge ${pos.side === 'LONG' ? 'badge-success' : 'badge-danger'}`}>
                                            {pos.side}
                                        </span>
                                    </td>
                                    <td>${pos.entryPrice.toLocaleString()}</td>
                                    <td>${pos.exitPrice.toLocaleString()}</td>
                                    <td className={pos.realizedPnL >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]'}>
                                        {pos.realizedPnL >= 0 ? '+' : ''}${pos.realizedPnL.toFixed(2)}
                                    </td>
                                    <td className="text-[#8b98a5]">{pos.exitReason}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* No Manual Close Notice */}
            <div className="card bg-[#1a1f26] mt-6">
                <div className="flex items-center gap-4 text-sm text-[#8b98a5]">
                    <span className="text-xl">🔒</span>
                    <span>
                        Manual position closing is disabled to maintain trading discipline.
                        All exits are handled by the system according to predefined rules.
                    </span>
                </div>
            </div>
        </div>
    );
}
