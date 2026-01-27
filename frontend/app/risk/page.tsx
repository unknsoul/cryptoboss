import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

// Mock risk data
const mockRisk = {
    dailyLoss: { current: -125, limit: -500, pct: 25 },
    weeklyLoss: { current: -280, limit: -1500, pct: 18.7 },
    tradeBudget: { daily: { used: 4, max: 10 }, perContext: { used: 1, max: 3 }, perBias: { used: 0, max: 2 } },
    allocation: {
        current: 'RANGING',
        pct: 75,
        contexts: [
            { name: 'TRENDING_UP', pct: 100 },
            { name: 'TRENDING_DOWN', pct: 100 },
            { name: 'RANGING', pct: 75, active: true },
            { name: 'HIGH_VOLATILITY', pct: 30 },
            { name: 'NO_TRADE', pct: 0 },
        ]
    },
    killSwitch: false,
    consecutiveLosses: 1,
};

export default function RiskCapitalPage() {
    return (
        <div>
            <PageHeader
                title="Risk & Capital"
                description="Make risk impossible to ignore"
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Daily Loss Limit */}
                <Card title="Daily Loss Limit">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Current Loss</span>
                            <span className={`text-xl font-bold ${mockRisk.dailyLoss.current < 0 ? 'text-red-400' : 'text-green-400'}`}>
                                ${mockRisk.dailyLoss.current}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Limit</span>
                            <span className="text-white font-medium">${mockRisk.dailyLoss.limit}</span>
                        </div>
                        <div className="h-3 bg-[#242b33] rounded-full overflow-hidden">
                            <div
                                className={`h-full transition-all ${mockRisk.dailyLoss.pct < 50 ? 'bg-green-500' :
                                        mockRisk.dailyLoss.pct < 75 ? 'bg-yellow-500' : 'bg-red-500'
                                    }`}
                                style={{ width: `${mockRisk.dailyLoss.pct}%` }}
                            />
                        </div>
                        <div className="text-center text-sm text-[#8b98a5]">
                            {mockRisk.dailyLoss.pct.toFixed(0)}% of limit used
                        </div>
                    </div>
                </Card>

                {/* Trade Budget */}
                <Card title="Trade Budget Counters">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Daily Trades</span>
                            <Badge variant={mockRisk.tradeBudget.daily.used < mockRisk.tradeBudget.daily.max ? 'success' : 'danger'}>
                                {mockRisk.tradeBudget.daily.used} / {mockRisk.tradeBudget.daily.max}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Per Context</span>
                            <Badge variant={mockRisk.tradeBudget.perContext.used < mockRisk.tradeBudget.perContext.max ? 'success' : 'warning'}>
                                {mockRisk.tradeBudget.perContext.used} / {mockRisk.tradeBudget.perContext.max}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Per Bias (Losses)</span>
                            <Badge variant={mockRisk.tradeBudget.perBias.used < mockRisk.tradeBudget.perBias.max ? 'success' : 'danger'}>
                                {mockRisk.tradeBudget.perBias.used} / {mockRisk.tradeBudget.perBias.max}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Consecutive Losses</span>
                            <Badge variant={mockRisk.consecutiveLosses < 2 ? 'success' : 'warning'}>
                                {mockRisk.consecutiveLosses}
                            </Badge>
                        </div>
                    </div>
                </Card>

                {/* Capital Allocation by Context */}
                <Card title="Capital Allocation by Context" className="lg:col-span-2">
                    <div className="space-y-3">
                        {mockRisk.allocation.contexts.map((ctx) => (
                            <div key={ctx.name} className="flex items-center gap-4">
                                <div className="w-36">
                                    <Badge variant={ctx.active ? 'info' : 'neutral'}>
                                        {ctx.name}
                                    </Badge>
                                </div>
                                <div className="flex-1 h-2 bg-[#242b33] rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${ctx.active ? 'bg-blue-500' : 'bg-[#3d4654]'}`}
                                        style={{ width: `${ctx.pct}%` }}
                                    />
                                </div>
                                <div className="w-12 text-right">
                                    <span className={`text-sm font-medium ${ctx.active ? 'text-white' : 'text-[#8b98a5]'}`}>
                                        {ctx.pct}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>

                {/* Kill Switch Status */}
                <Card title="Kill Switch" className="lg:col-span-2">
                    <div className="flex items-center justify-between p-4 bg-[#242b33] rounded-lg">
                        <div className="flex items-center gap-4">
                            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${mockRisk.killSwitch ? 'bg-red-500/20' : 'bg-green-500/20'
                                }`}>
                                {mockRisk.killSwitch ? (
                                    <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                ) : (
                                    <svg className="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                )}
                            </div>
                            <div>
                                <span className="text-white font-medium">
                                    {mockRisk.killSwitch ? 'KILL SWITCH ACTIVE' : 'System Normal'}
                                </span>
                                <p className="text-sm text-[#8b98a5]">
                                    {mockRisk.killSwitch
                                        ? 'All trading halted. Manual recovery required.'
                                        : 'Trading operations enabled.'}
                                </p>
                            </div>
                        </div>
                        <Badge variant={mockRisk.killSwitch ? 'danger' : 'success'} size="md">
                            {mockRisk.killSwitch ? 'HALTED' : 'ACTIVE'}
                        </Badge>
                    </div>
                </Card>
            </div>
        </div>
    );
}
