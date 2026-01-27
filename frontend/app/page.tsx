import { Card } from '../components/shared/Card';
import { Badge } from '../components/shared/Badge';
import { StatusDot } from '../components/shared/StatusDot';
import { PageHeader } from '../components/layout/PageHeader';

// Simulated data - will be replaced with API calls
const mockData = {
    context: { regime: 'RANGING', confidence: 0.72, tradingAllowed: true },
    bias: { direction: 'LONG_BIAS', conviction: 0.68 },
    health: { stage: 'NORMAL', score: 0.95 },
    capital: { allocation: 0.75, available: 7500 },
    pnl: { daily: 125.50, dailyPct: 1.26 },
    drawdown: { current: 2.3, max: 10 },
};

export default function OverviewPage() {
    return (
        <div>
            <PageHeader
                title="Overview"
                description="High-level system health snapshot"
            />

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Market Context Widget */}
                <Card title="Market Context">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Regime</span>
                            <Badge variant="info">{mockData.context.regime}</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Confidence</span>
                            <span className="text-white font-medium">{(mockData.context.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Trading</span>
                            <Badge variant={mockData.context.tradingAllowed ? 'success' : 'danger'}>
                                {mockData.context.tradingAllowed ? 'ALLOWED' : 'BLOCKED'}
                            </Badge>
                        </div>
                    </div>
                </Card>

                {/* Bias Widget */}
                <Card title="Current Bias">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Direction</span>
                            <Badge variant={
                                mockData.bias.direction.includes('LONG') ? 'success' :
                                    mockData.bias.direction.includes('SHORT') ? 'danger' : 'neutral'
                            }>
                                {mockData.bias.direction.replace('_', ' ')}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Conviction</span>
                            <span className="text-white font-medium">{(mockData.bias.conviction * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-green-500 transition-all"
                                style={{ width: `${mockData.bias.conviction * 100}%` }}
                            />
                        </div>
                    </div>
                </Card>

                {/* Exchange Health Widget */}
                <Card title="Exchange Health">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Stage</span>
                            <Badge variant={
                                mockData.health.stage === 'NORMAL' ? 'success' :
                                    mockData.health.stage === 'DEGRADED' ? 'warning' : 'danger'
                            }>
                                {mockData.health.stage}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Health Score</span>
                            <span className="text-white font-medium">{(mockData.health.score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-green-500 transition-all"
                                style={{ width: `${mockData.health.score * 100}%` }}
                            />
                        </div>
                    </div>
                </Card>

                {/* Capital Allocation Widget */}
                <Card title="Capital Allocation">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Allocation</span>
                            <span className="text-white font-medium">{(mockData.capital.allocation * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Available</span>
                            <span className="text-white font-medium">${mockData.capital.available.toLocaleString()}</span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all"
                                style={{ width: `${mockData.capital.allocation * 100}%` }}
                            />
                        </div>
                    </div>
                </Card>

                {/* Daily PnL Widget */}
                <Card title="Daily P&L">
                    <div className="space-y-4">
                        <div className="text-center">
                            <span className={`text-3xl font-bold ${mockData.pnl.daily >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {mockData.pnl.daily >= 0 ? '+' : ''}{mockData.pnl.dailyPct.toFixed(2)}%
                            </span>
                            <p className="text-[#8b98a5] text-sm mt-1">
                                ${mockData.pnl.daily >= 0 ? '+' : ''}{mockData.pnl.daily.toFixed(2)}
                            </p>
                        </div>
                    </div>
                </Card>

                {/* Drawdown Gauge Widget */}
                <Card title="Drawdown">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Current</span>
                            <span className={`font-medium ${mockData.drawdown.current < 3 ? 'text-green-400' :
                                    mockData.drawdown.current < 5 ? 'text-yellow-400' : 'text-red-400'
                                }`}>
                                {mockData.drawdown.current.toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Max Allowed</span>
                            <span className="text-white font-medium">{mockData.drawdown.max}%</span>
                        </div>
                        <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className={`h-full transition-all ${mockData.drawdown.current / mockData.drawdown.max < 0.3 ? 'bg-green-500' :
                                        mockData.drawdown.current / mockData.drawdown.max < 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                                    }`}
                                style={{ width: `${(mockData.drawdown.current / mockData.drawdown.max) * 100}%` }}
                            />
                        </div>
                    </div>
                </Card>
            </div>

            {/* System Status Banner */}
            <div className="mt-6">
                <Card>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <StatusDot status="success" pulse />
                            <span className="text-white font-medium">System Operating Normally</span>
                        </div>
                        <span className="text-[#8b98a5] text-sm">
                            Last update: {new Date().toLocaleTimeString()}
                        </span>
                    </div>
                </Card>
            </div>
        </div>
    );
}
