import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

// Mock data - will be replaced with WebSocket data
const mockData = {
    price: { symbol: 'BTC/USDT', current: 42150.50, change24h: 2.34 },
    positions: [
        { id: 1, symbol: 'BTC/USDT', side: 'LONG', size: 0.25, entry: 41000, current: 42150, pnl: 287.50, pnlPct: 2.80 },
    ],
    proposals: { active: 2, pending: 1 },
    execution: { state: 'READY', lastOrder: null },
};

export default function LiveStatusPage() {
    return (
        <div>
            <PageHeader
                title="Live Status"
                description="Real-time monitoring of trading activity"
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Price Feed */}
                <Card title="Live Price" subtitle="Real-time market data">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-2xl font-bold text-white">{mockData.price.symbol}</span>
                            <Badge variant={mockData.price.change24h >= 0 ? 'success' : 'danger'}>
                                {mockData.price.change24h >= 0 ? '+' : ''}{mockData.price.change24h.toFixed(2)}%
                            </Badge>
                        </div>
                        <div className="text-4xl font-bold text-white">
                            ${mockData.price.current.toLocaleString()}
                        </div>
                        <div className="text-sm text-[#8b98a5]">
                            Last update: {new Date().toLocaleTimeString()}
                        </div>
                    </div>
                </Card>

                {/* Execution State */}
                <Card title="Execution State" subtitle="Current system readiness">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">State</span>
                            <Badge variant={mockData.execution.state === 'READY' ? 'success' : 'warning'}>
                                {mockData.execution.state}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Active Proposals</span>
                            <span className="text-white font-medium">{mockData.proposals.active}</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Pending</span>
                            <span className="text-white font-medium">{mockData.proposals.pending}</span>
                        </div>
                    </div>
                </Card>

                {/* Open Positions */}
                <Card title="Open Positions" className="lg:col-span-2" noPadding>
                    {mockData.positions.length > 0 ? (
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-[#2d3640]">
                                        <th className="text-left text-xs text-[#8b98a5] font-medium py-3 px-4">Symbol</th>
                                        <th className="text-left text-xs text-[#8b98a5] font-medium py-3 px-4">Side</th>
                                        <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">Size</th>
                                        <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">Entry</th>
                                        <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">Current</th>
                                        <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">P&L</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {mockData.positions.map((pos) => (
                                        <tr key={pos.id} className="border-b border-[#2d3640] hover:bg-[#1a1f26]">
                                            <td className="py-3 px-4 text-white">{pos.symbol}</td>
                                            <td className="py-3 px-4">
                                                <Badge variant={pos.side === 'LONG' ? 'success' : 'danger'}>
                                                    {pos.side}
                                                </Badge>
                                            </td>
                                            <td className="py-3 px-4 text-right text-white">{pos.size}</td>
                                            <td className="py-3 px-4 text-right text-[#8b98a5]">${pos.entry.toLocaleString()}</td>
                                            <td className="py-3 px-4 text-right text-white">${pos.current.toLocaleString()}</td>
                                            <td className={`py-3 px-4 text-right font-medium ${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)} ({pos.pnlPct.toFixed(2)}%)
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div className="p-8 text-center text-[#8b98a5]">
                            No open positions
                        </div>
                    )}
                </Card>
            </div>
        </div>
    );
}
