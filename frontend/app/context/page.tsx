'use client';

/**
 * Market Context Page
 * 
 * Purpose: Market regime understanding
 * Rules:
 * - Timeline view
 * - No technical indicator clutter
 */

// Mock context data
const contextData = {
    current: {
        state: 'RANGING',
        confidence: 78,
        timeInState: '2h 34m',
        tradingAllowed: true,
    },
    cooldown: {
        active: false,
        remaining: '0m',
        reason: null,
    },
    history: [
        {
            state: 'RANGING',
            startTime: '12:00',
            duration: '2h 34m',
            active: true,
            transitionReason: 'Volatility decreased below threshold'
        },
        {
            state: 'TRENDING_UP',
            startTime: '09:30',
            duration: '2h 30m',
            active: false,
            transitionReason: 'Higher highs confirmed on 1H'
        },
        {
            state: 'HIGH_VOLATILITY',
            startTime: '08:45',
            duration: '45m',
            active: false,
            transitionReason: 'News event volatility spike'
        },
        {
            state: 'TRENDING_UP',
            startTime: '06:00',
            duration: '2h 45m',
            active: false,
            transitionReason: 'Session open momentum'
        },
    ],
};

const stateColors: Record<string, string> = {
    'TRENDING_UP': 'badge-success',
    'TRENDING_DOWN': 'badge-danger',
    'RANGING': 'badge-accent',
    'HIGH_VOLATILITY': 'badge-warning',
    'NO_TRADE': 'badge-neutral',
};

export default function MarketContextPage() {
    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Market Context</h1>
                <p className="text-[#8b98a5] text-sm">
                    Market regime understanding — no indicator clutter
                </p>
            </div>

            {/* Current State Card */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Current State</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div className="text-center">
                        <span className="label block mb-2">Regime</span>
                        <span className={`badge ${stateColors[contextData.current.state]} text-lg px-4 py-2`}>
                            {contextData.current.state}
                        </span>
                    </div>
                    <div className="text-center">
                        <span className="label block mb-2">Confidence</span>
                        <span className="value-xl">{contextData.current.confidence}%</span>
                    </div>
                    <div className="text-center">
                        <span className="label block mb-2">Time in State</span>
                        <span className="value-lg">{contextData.current.timeInState}</span>
                    </div>
                    <div className="text-center">
                        <span className="label block mb-2">Trading</span>
                        <span className={`badge ${contextData.current.tradingAllowed ? 'badge-success' : 'badge-danger'}`}>
                            {contextData.current.tradingAllowed ? 'ALLOWED' : 'BLOCKED'}
                        </span>
                    </div>
                </div>

                {/* Cooldown Timer */}
                {contextData.cooldown.active && (
                    <div className="mt-6 p-4 bg-[#c4a052]/10 rounded-md">
                        <div className="flex items-center gap-3">
                            <span className="text-[#c4a052]">⏱</span>
                            <div>
                                <span className="text-[#c4a052] font-medium">
                                    Transition Cooldown Active: {contextData.cooldown.remaining} remaining
                                </span>
                                <p className="text-sm text-[#8b98a5] mt-1">
                                    {contextData.cooldown.reason}
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Context History Timeline */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Context History (Today)</span>
                </div>

                <div className="timeline">
                    {contextData.history.map((item, idx) => (
                        <div
                            key={idx}
                            className={`timeline-item ${item.active ? 'timeline-item-success' : ''}`}
                        >
                            <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center gap-3">
                                    <span className="text-xs font-mono text-[#6b7280]">{item.startTime}</span>
                                    <span className={`badge ${stateColors[item.state]}`}>{item.state}</span>
                                    {item.active && (
                                        <span className="badge badge-success">CURRENT</span>
                                    )}
                                </div>
                                <span className="text-sm text-[#8b98a5]">{item.duration}</span>
                            </div>
                            <p className="text-sm text-[#6b7280]">
                                Transition: {item.transitionReason}
                            </p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Context State Descriptions */}
            <div className="card bg-[#1a1f26]">
                <div className="card-header">
                    <span className="card-title">State Reference</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-3 border border-[#2d3640] rounded-md">
                        <span className="badge badge-success mb-2">TRENDING_UP</span>
                        <p className="text-sm text-[#8b98a5]">
                            Clear upward momentum. Long entries enabled, shorts restricted.
                        </p>
                    </div>
                    <div className="p-3 border border-[#2d3640] rounded-md">
                        <span className="badge badge-danger mb-2">TRENDING_DOWN</span>
                        <p className="text-sm text-[#8b98a5]">
                            Clear downward momentum. Short entries enabled, longs restricted.
                        </p>
                    </div>
                    <div className="p-3 border border-[#2d3640] rounded-md">
                        <span className="badge badge-accent mb-2">RANGING</span>
                        <p className="text-sm text-[#8b98a5]">
                            Consolidation phase. Mean reversion strategies enabled, reduced position sizes.
                        </p>
                    </div>
                    <div className="p-3 border border-[#2d3640] rounded-md">
                        <span className="badge badge-warning mb-2">HIGH_VOLATILITY</span>
                        <p className="text-sm text-[#8b98a5]">
                            Elevated volatility. Trading restricted, capital protection mode.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
