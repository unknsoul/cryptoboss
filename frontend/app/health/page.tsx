'use client';

/**
 * Execution & Health Page
 * 
 * Purpose: Infrastructure reliability monitoring
 * Rules:
 * - Escalation stages clearly labeled
 * - Manual recovery actions visible
 */

// Mock execution health data
const healthData = {
    exchange: {
        latency: 45,
        latencyHistory: [42, 48, 45, 52, 41, 45],
        status: 'NORMAL' as const,
    },
    orders: {
        rejectionRate: 0.8,
        partialFillRatio: 2.1,
        avgFillTime: 120,
    },
    escalation: {
        currentStage: 0,
        stages: [
            { level: 0, name: 'NORMAL', description: 'All systems operational', color: 'success' },
            { level: 1, name: 'DEGRADED', description: 'Increased latency or minor issues', color: 'warning' },
            { level: 2, name: 'CLOSE_ONLY', description: 'New entries blocked, exits only', color: 'warning' },
            { level: 3, name: 'HALTED', description: 'All trading suspended', color: 'danger' },
        ],
    },
    recentEvents: [
        { time: '14:32:15', event: 'Order filled', details: 'BTC/USDT LONG 0.025 @ 89168.42', type: 'success' },
        { time: '14:28:00', event: 'Latency spike', details: '156ms (above 100ms threshold)', type: 'warning' },
        { time: '14:15:00', event: 'Connection restored', details: 'Binance WebSocket reconnected', type: 'success' },
        { time: '14:14:30', event: 'Connection lost', details: 'WebSocket disconnected', type: 'danger' },
    ],
};

function HealthMetric({ label, value, unit, status }: {
    label: string;
    value: string | number;
    unit?: string;
    status?: 'good' | 'warning' | 'bad';
}) {
    const statusColors = {
        good: 'text-[#4a9268]',
        warning: 'text-[#c4a052]',
        bad: 'text-[#a65454]',
    };

    return (
        <div className="bg-[#1a1f26] rounded-md p-4 text-center">
            <span className="label block">{label}</span>
            <span className={`value-lg block mt-1 ${status ? statusColors[status] : 'text-[#e7e9ea]'}`}>
                {value}{unit && <span className="text-sm text-[#8b98a5] ml-1">{unit}</span>}
            </span>
        </div>
    );
}

export default function ExecutionHealthPage() {
    const currentStage = healthData.escalation.stages[healthData.escalation.currentStage];

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Execution & Health</h1>
                <p className="text-[#8b98a5] text-sm">
                    Infrastructure reliability monitoring
                </p>
            </div>

            {/* Current Escalation Stage */}
            <div className={`card border-l-4 ${currentStage.color === 'success' ? 'border-l-[#4a9268]' :
                    currentStage.color === 'warning' ? 'border-l-[#c4a052]' : 'border-l-[#a65454]'
                }`}>
                <div className="flex items-center justify-between">
                    <div>
                        <span className="label">Current Stage</span>
                        <div className="flex items-center gap-3 mt-2">
                            <span className={`badge badge-${currentStage.color}`}>
                                {currentStage.name}
                            </span>
                            <span className="text-[#e7e9ea]">{currentStage.description}</span>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        {healthData.escalation.stages.map((stage, idx) => (
                            <div
                                key={stage.level}
                                className={`w-3 h-3 rounded-full ${idx === healthData.escalation.currentStage
                                        ? stage.color === 'success' ? 'bg-[#4a9268]' :
                                            stage.color === 'warning' ? 'bg-[#c4a052]' : 'bg-[#a65454]'
                                        : 'bg-[#2d3640]'
                                    }`}
                                title={stage.name}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <HealthMetric
                    label="Latency"
                    value={healthData.exchange.latency}
                    unit="ms"
                    status={healthData.exchange.latency < 100 ? 'good' :
                        healthData.exchange.latency < 300 ? 'warning' : 'bad'}
                />
                <HealthMetric
                    label="Order Rejection Rate"
                    value={healthData.orders.rejectionRate}
                    unit="%"
                    status={healthData.orders.rejectionRate < 2 ? 'good' :
                        healthData.orders.rejectionRate < 5 ? 'warning' : 'bad'}
                />
                <HealthMetric
                    label="Partial Fill Ratio"
                    value={healthData.orders.partialFillRatio}
                    unit="%"
                    status={healthData.orders.partialFillRatio < 5 ? 'good' :
                        healthData.orders.partialFillRatio < 10 ? 'warning' : 'bad'}
                />
                <HealthMetric
                    label="Avg Fill Time"
                    value={healthData.orders.avgFillTime}
                    unit="ms"
                    status={healthData.orders.avgFillTime < 200 ? 'good' :
                        healthData.orders.avgFillTime < 500 ? 'warning' : 'bad'}
                />
            </div>

            {/* Escalation Stages Reference */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Escalation Stages</span>
                </div>

                <div className="space-y-3">
                    {healthData.escalation.stages.map((stage) => (
                        <div
                            key={stage.level}
                            className={`flex items-center justify-between p-3 rounded-md ${stage.level === healthData.escalation.currentStage
                                    ? 'bg-[#1a1f26] border border-[#2d3640]'
                                    : ''
                                }`}
                        >
                            <div className="flex items-center gap-4">
                                <span className="text-[#6b7280] font-mono">L{stage.level}</span>
                                <span className={`badge badge-${stage.color}`}>{stage.name}</span>
                                <span className="text-[#8b98a5] text-sm">{stage.description}</span>
                            </div>
                            {stage.level === healthData.escalation.currentStage && (
                                <span className="badge badge-neutral">CURRENT</span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Recent Events Log */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Recent Events</span>
                </div>

                <div className="space-y-2">
                    {healthData.recentEvents.map((event, idx) => (
                        <div
                            key={idx}
                            className="flex items-center gap-4 py-2 border-b border-[#2d3640] last:border-0"
                        >
                            <span className="text-xs font-mono text-[#6b7280] w-20">{event.time}</span>
                            <div className={`status-dot ${event.type === 'success' ? 'status-dot-healthy' :
                                    event.type === 'warning' ? 'status-dot-warning' : 'status-dot-critical'
                                }`} />
                            <span className="text-[#e7e9ea] text-sm">{event.event}</span>
                            <span className="text-[#8b98a5] text-sm">{event.details}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
