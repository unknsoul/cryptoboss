import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

const mockPositions = [
    {
        id: 1,
        symbol: 'BTC/USDT',
        side: 'LONG',
        size: 0.25,
        entry: 41000,
        current: 42150,
        stopLoss: 39500,
        takeProfit: 45000,
        pnl: 287.50,
        pnlPct: 2.80,
        risk: 375,
        riskPct: 3.66,
        entryReason: 'RANGING context, LONG_BIAS (68%), DCA strategy (score=0.805)',
        timestamp: '2026-01-27T20:15:00',
    },
];

export default function PositionsPage() {
    return (
        <div>
            <PageHeader
                title="Positions"
                description="Trade-level visibility with entry reasoning"
            />

            {mockPositions.length > 0 ? (
                <div className="space-y-4">
                    {mockPositions.map((pos) => (
                        <Card key={pos.id}>
                            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                                {/* Position Info */}
                                <div className="space-y-3">
                                    <div className="flex items-center gap-3">
                                        <span className="text-xl font-bold text-white">{pos.symbol}</span>
                                        <Badge variant={pos.side === 'LONG' ? 'success' : 'danger'}>
                                            {pos.side}
                                        </Badge>
                                    </div>
                                    <div className="text-sm text-[#8b98a5]">
                                        Size: <span className="text-white">{pos.size} BTC</span>
                                    </div>
                                    <div className="text-sm text-[#8b98a5]">
                                        Opened: {new Date(pos.timestamp).toLocaleString()}
                                    </div>
                                </div>

                                {/* Price Info */}
                                <div className="space-y-3">
                                    <div className="text-sm text-[#8b98a5]">Entry</div>
                                    <div className="text-lg text-white">${pos.entry.toLocaleString()}</div>
                                    <div className="text-sm text-[#8b98a5]">
                                        Current: <span className="text-white">${pos.current.toLocaleString()}</span>
                                    </div>
                                </div>

                                {/* P&L */}
                                <div className="space-y-3">
                                    <div className="text-sm text-[#8b98a5]">Unrealized P&L</div>
                                    <div className={`text-2xl font-bold ${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}
                                    </div>
                                    <div className={`text-sm ${pos.pnlPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {pos.pnlPct >= 0 ? '+' : ''}{pos.pnlPct.toFixed(2)}%
                                    </div>
                                </div>

                                {/* Risk */}
                                <div className="space-y-3">
                                    <div className="text-sm text-[#8b98a5]">Risk Exposure</div>
                                    <div className="text-lg text-white">${pos.risk} ({pos.riskPct.toFixed(1)}%)</div>
                                    <div className="flex gap-2 text-sm">
                                        <span className="text-red-400">SL: ${pos.stopLoss.toLocaleString()}</span>
                                        <span className="text-green-400">TP: ${pos.takeProfit.toLocaleString()}</span>
                                    </div>
                                </div>
                            </div>

                            {/* Entry Reasoning */}
                            <div className="mt-6 pt-4 border-t border-[#2d3640]">
                                <div className="text-sm text-[#8b98a5] mb-2">Entry Reasoning</div>
                                <div className="p-3 bg-[#242b33] rounded-lg text-sm text-white">
                                    {pos.entryReason}
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>
            ) : (
                <Card>
                    <div className="text-center py-12 text-[#8b98a5]">
                        <p className="text-lg">No open positions</p>
                        <p className="text-sm mt-2">The system will open positions when conditions are met.</p>
                    </div>
                </Card>
            )}
        </div>
    );
}
