import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

const mockStrategies = [
    {
        id: 'dca_btc',
        name: 'DCA Bitcoin',
        enabled: true,
        healthScore: 0.85,
        recentDecay: 0.92,
        wins: 12,
        losses: 3,
        lastTrade: '2026-01-27T20:15:00',
    },
    {
        id: 'grid_btc',
        name: 'Grid Trading BTC',
        enabled: true,
        healthScore: 0.72,
        recentDecay: 0.85,
        wins: 8,
        losses: 4,
        lastTrade: '2026-01-27T18:30:00',
    },
    {
        id: 'scalp_eth',
        name: 'Scalp ETH',
        enabled: false,
        healthScore: 0.45,
        recentDecay: 0.60,
        wins: 5,
        losses: 7,
        lastTrade: '2026-01-26T14:00:00',
    },
];

export default function StrategiesPage() {
    return (
        <div>
            <PageHeader
                title="Strategies"
                description="Control without micromanagement – no manual entry buttons"
            />

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {mockStrategies.map((strategy) => (
                    <Card key={strategy.id}>
                        <div className="space-y-4">
                            {/* Header */}
                            <div className="flex items-center justify-between">
                                <span className="text-lg font-medium text-white">{strategy.name}</span>
                                <Badge variant={strategy.enabled ? 'success' : 'neutral'}>
                                    {strategy.enabled ? 'ENABLED' : 'DISABLED'}
                                </Badge>
                            </div>

                            {/* Health Score */}
                            <div>
                                <div className="flex items-center justify-between text-sm mb-1">
                                    <span className="text-[#8b98a5]">Health Score</span>
                                    <span className={`font-medium ${strategy.healthScore >= 0.7 ? 'text-green-400' :
                                            strategy.healthScore >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                                        }`}>
                                        {(strategy.healthScore * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${strategy.healthScore >= 0.7 ? 'bg-green-500' :
                                                strategy.healthScore >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                                            }`}
                                        style={{ width: `${strategy.healthScore * 100}%` }}
                                    />
                                </div>
                            </div>

                            {/* Recent Performance Decay */}
                            <div>
                                <div className="flex items-center justify-between text-sm mb-1">
                                    <span className="text-[#8b98a5]">Recent Performance</span>
                                    <span className="font-medium text-white">
                                        {(strategy.recentDecay * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div className="h-2 bg-[#242b33] rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-blue-500"
                                        style={{ width: `${strategy.recentDecay * 100}%` }}
                                    />
                                </div>
                            </div>

                            {/* Win/Loss */}
                            <div className="flex items-center justify-between text-sm">
                                <span className="text-[#8b98a5]">Win/Loss</span>
                                <span>
                                    <span className="text-green-400">{strategy.wins}W</span>
                                    {' / '}
                                    <span className="text-red-400">{strategy.losses}L</span>
                                </span>
                            </div>

                            {/* Last Trade */}
                            <div className="text-xs text-[#6b7280]">
                                Last trade: {new Date(strategy.lastTrade).toLocaleString()}
                            </div>
                        </div>
                    </Card>
                ))}
            </div>

            {/* Info Notice */}
            <div className="mt-6 p-4 bg-[#1a1f26] rounded-lg border border-[#2d3640]">
                <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                        <p className="text-sm text-white font-medium">Strategies propose, they don't execute</p>
                        <p className="text-xs text-[#8b98a5] mt-1">
                            All trades go through the 9-stage execution flow. No manual trade buttons available.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
