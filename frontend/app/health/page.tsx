import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

const mockHealth = {
    stage: 'NORMAL',
    latency: { avg: 145, warning: 500, critical: 2000 },
    rejectionRate: { current: 2.5, warning: 10, critical: 30 },
    partialFills: { rate: 8, warning: 20, critical: 50 },
    wsLag: { current: 85, warning: 1000, critical: 5000 },
    consecutiveFailures: 0,
    lastOrder: '2026-01-27T22:15:00',
};

const stages = [
    { name: 'NORMAL', color: 'success', description: 'Full trading capacity' },
    { name: 'DEGRADED_REDUCED', color: 'warning', description: '50% size reduction' },
    { name: 'DEGRADED_CLOSE', color: 'warning', description: 'Close-only mode' },
    { name: 'HALTED', color: 'danger', description: 'All trading stopped' },
];

export default function HealthPage() {
    return (
        <div>
            <PageHeader
                title="Execution & Health"
                description="Infrastructure reliability monitoring"
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Current Stage */}
                <Card title="Escalation Stage">
                    <div className="space-y-4">
                        {stages.map((stage) => (
                            <div
                                key={stage.name}
                                className={`p-3 rounded-lg border ${mockHealth.stage === stage.name
                                        ? 'border-green-500 bg-green-500/10'
                                        : 'border-[#2d3640] bg-[#242b33]'
                                    }`}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-3 h-3 rounded-full ${mockHealth.stage === stage.name ? 'bg-green-500' : 'bg-[#3d4654]'
                                            }`} />
                                        <span className={`font-medium ${mockHealth.stage === stage.name ? 'text-white' : 'text-[#8b98a5]'
                                            }`}>
                                            {stage.name}
                                        </span>
                                    </div>
                                    {mockHealth.stage === stage.name && (
                                        <Badge variant="success">CURRENT</Badge>
                                    )}
                                </div>
                                <p className="text-xs text-[#6b7280] mt-1 ml-6">{stage.description}</p>
                            </div>
                        ))}
                    </div>
                </Card>

                {/* Metrics Grid */}
                <div className="space-y-4">
                    {/* Latency */}
                    <Card>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-[#8b98a5]">API Latency</span>
                            <span className={`font-medium ${mockHealth.latency.avg < mockHealth.latency.warning ? 'text-green-400' : 'text-yellow-400'
                                }`}>
                                {mockHealth.latency.avg}ms
                            </span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-green-500"
                                style={{ width: `${(mockHealth.latency.avg / mockHealth.latency.warning) * 100}%` }}
                            />
                        </div>
                        <div className="flex justify-between text-xs text-[#6b7280] mt-1">
                            <span>0</span>
                            <span>Warning: {mockHealth.latency.warning}ms</span>
                        </div>
                    </Card>

                    {/* Rejection Rate */}
                    <Card>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-[#8b98a5]">Rejection Rate</span>
                            <span className={`font-medium ${mockHealth.rejectionRate.current < mockHealth.rejectionRate.warning ? 'text-green-400' : 'text-yellow-400'
                                }`}>
                                {mockHealth.rejectionRate.current}%
                            </span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-green-500"
                                style={{ width: `${(mockHealth.rejectionRate.current / mockHealth.rejectionRate.warning) * 100}%` }}
                            />
                        </div>
                    </Card>

                    {/* WebSocket Lag */}
                    <Card>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-[#8b98a5]">WebSocket Lag</span>
                            <span className={`font-medium ${mockHealth.wsLag.current < mockHealth.wsLag.warning ? 'text-green-400' : 'text-yellow-400'
                                }`}>
                                {mockHealth.wsLag.current}ms
                            </span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-green-500"
                                style={{ width: `${(mockHealth.wsLag.current / mockHealth.wsLag.warning) * 100}%` }}
                            />
                        </div>
                    </Card>
                </div>

                {/* Last Order Info */}
                <Card title="Last Order" className="lg:col-span-2">
                    <div className="flex items-center justify-between">
                        <span className="text-[#8b98a5]">Timestamp</span>
                        <span className="text-white">{new Date(mockHealth.lastOrder).toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between mt-3">
                        <span className="text-[#8b98a5]">Consecutive Failures</span>
                        <Badge variant={mockHealth.consecutiveFailures === 0 ? 'success' : 'warning'}>
                            {mockHealth.consecutiveFailures}
                        </Badge>
                    </div>
                </Card>
            </div>
        </div>
    );
}
