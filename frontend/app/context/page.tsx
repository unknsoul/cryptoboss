import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

const mockContext = {
    current: { state: 'RANGING', since: '2026-01-27T19:30:00', duration: 3.2 },
    history: [
        { state: 'RANGING', from: '2026-01-27T19:30:00', to: null, duration: 3.2 },
        { state: 'TRENDING_UP', from: '2026-01-27T15:00:00', to: '2026-01-27T19:30:00', duration: 4.5 },
        { state: 'HIGH_VOLATILITY', from: '2026-01-27T13:00:00', to: '2026-01-27T15:00:00', duration: 2.0 },
        { state: 'RANGING', from: '2026-01-27T08:00:00', to: '2026-01-27T13:00:00', duration: 5.0 },
    ],
    cooldown: { active: false, remaining: 0 },
    minDuration: 2,
};

const stateColors: Record<string, string> = {
    'TRENDING_UP': 'success',
    'TRENDING_DOWN': 'danger',
    'RANGING': 'info',
    'HIGH_VOLATILITY': 'warning',
    'NO_TRADE': 'neutral',
};

export default function ContextPage() {
    return (
        <div>
            <PageHeader
                title="Market Context"
                description="Macro understanding of current market state"
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Current State */}
                <Card title="Current State">
                    <div className="space-y-4">
                        <div className="flex items-center justify-center py-4">
                            <Badge variant={stateColors[mockContext.current.state] as any} size="md">
                                {mockContext.current.state}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Duration</span>
                            <span className="text-white font-medium">{mockContext.current.duration.toFixed(1)}h</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Min Required</span>
                            <span className="text-white font-medium">{mockContext.minDuration}h</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Since</span>
                            <span className="text-white text-sm">
                                {new Date(mockContext.current.since).toLocaleTimeString()}
                            </span>
                        </div>
                    </div>
                </Card>

                {/* Transition Cooldown */}
                <Card title="Transition Status">
                    <div className="space-y-4">
                        {mockContext.cooldown.active ? (
                            <>
                                <div className="text-center">
                                    <span className="text-2xl font-bold text-yellow-400">
                                        {mockContext.cooldown.remaining}m
                                    </span>
                                    <p className="text-sm text-[#8b98a5] mt-1">Cooldown remaining</p>
                                </div>
                                <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                                    <div className="h-full bg-yellow-500 animate-pulse" style={{ width: '60%' }} />
                                </div>
                            </>
                        ) : (
                            <div className="text-center py-4">
                                <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center mx-auto">
                                    <svg className="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                </div>
                                <p className="text-sm text-[#8b98a5] mt-3">Transition allowed</p>
                            </div>
                        )}
                    </div>
                </Card>

                {/* State Machine Status */}
                <Card title="State Machine">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Valid Since</span>
                            <Badge variant="success">YES</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Trading Allowed</span>
                            <Badge variant={mockContext.current.state !== 'NO_TRADE' ? 'success' : 'danger'}>
                                {mockContext.current.state !== 'NO_TRADE' ? 'YES' : 'NO'}
                            </Badge>
                        </div>
                    </div>
                </Card>

                {/* Context History */}
                <Card title="Context History" subtitle="Last 24 hours" className="lg:col-span-3">
                    <div className="space-y-3">
                        {mockContext.history.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-4 p-3 bg-[#242b33] rounded-lg">
                                <Badge variant={stateColors[item.state] as any}>{item.state}</Badge>
                                <div className="flex-1 text-sm text-[#8b98a5]">
                                    {new Date(item.from).toLocaleTimeString()}
                                    {item.to ? ` → ${new Date(item.to).toLocaleTimeString()}` : ' → now'}
                                </div>
                                <span className="text-sm text-white font-medium">{item.duration.toFixed(1)}h</span>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>
        </div>
    );
}
