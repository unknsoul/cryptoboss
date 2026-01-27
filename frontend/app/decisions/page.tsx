import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

// Mock decision flow data
const mockDecisions = [
    {
        id: 1,
        timestamp: '2026-01-27T22:45:00',
        stage: 'MARKET_CONTEXT',
        result: 'PASS',
        reason: 'Context: RANGING, trading allowed',
    },
    {
        id: 2,
        timestamp: '2026-01-27T22:45:01',
        stage: 'BIAS_ENGINE',
        result: 'PASS',
        reason: 'Bias: LONG_BIAS (68% conviction)',
    },
    {
        id: 3,
        timestamp: '2026-01-27T22:45:02',
        stage: 'BIAS_PRE_FILTER',
        result: 'PASS',
        reason: '2 proposals passed, 1 filtered (direction mismatch)',
    },
    {
        id: 4,
        timestamp: '2026-01-27T22:45:03',
        stage: 'SCORING_CONTRACT',
        result: 'PASS',
        reason: 'Validated: dca_btc (score=0.805)',
    },
    {
        id: 5,
        timestamp: '2026-01-27T22:45:04',
        stage: 'CAPITAL_GOVERNOR',
        result: 'PASS',
        reason: 'Allocated: 75% ($7,500 available)',
    },
];

const mockRejections = [
    { id: 1, strategy: 'grid_btc', reason: 'Direction SHORT conflicts with LONG_BIAS', stage: 'BIAS_PRE_FILTER' },
    { id: 2, strategy: 'scalp_eth', reason: 'Score 0.42 below threshold 0.50', stage: 'SCORING_CONTRACT' },
];

export default function DecisionFlowPage() {
    return (
        <div>
            <PageHeader
                title="Decision Flow"
                description="Understand WHY the bot acted or did nothing"
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Decision Timeline */}
                <Card title="Execution Flow" subtitle="9-stage pipeline" className="lg:col-span-2">
                    <div className="space-y-4">
                        {mockDecisions.map((decision, idx) => (
                            <div key={decision.id} className="flex items-start gap-4">
                                {/* Timeline connector */}
                                <div className="flex flex-col items-center">
                                    <div className={`w-3 h-3 rounded-full ${decision.result === 'PASS' ? 'bg-green-500' : 'bg-red-500'
                                        }`} />
                                    {idx < mockDecisions.length - 1 && (
                                        <div className="w-0.5 h-8 bg-[#2d3640] mt-1" />
                                    )}
                                </div>

                                {/* Content */}
                                <div className="flex-1 pb-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-white">{decision.stage}</span>
                                        <Badge variant={decision.result === 'PASS' ? 'success' : 'danger'} size="sm">
                                            {decision.result}
                                        </Badge>
                                    </div>
                                    <p className="text-sm text-[#8b98a5] mt-1">{decision.reason}</p>
                                    <span className="text-xs text-[#6b7280] mt-1 block">
                                        {new Date(decision.timestamp).toLocaleTimeString()}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>

                {/* Rejection Reasons */}
                <Card title="Recent Rejections" subtitle="Why proposals failed">
                    {mockRejections.length > 0 ? (
                        <div className="space-y-4">
                            {mockRejections.map((rejection) => (
                                <div key={rejection.id} className="p-3 bg-[#242b33] rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-white">{rejection.strategy}</span>
                                        <Badge variant="neutral" size="sm">{rejection.stage}</Badge>
                                    </div>
                                    <p className="text-xs text-[#8b98a5]">{rejection.reason}</p>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-[#8b98a5] text-center py-4">No recent rejections</p>
                    )}
                </Card>

                {/* Capital Veto Messages */}
                <Card title="Capital Decisions" className="lg:col-span-3">
                    <div className="flex items-center gap-4 p-4 bg-[#242b33] rounded-lg">
                        <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                            <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                        </div>
                        <div>
                            <span className="text-white font-medium">Capital Approved</span>
                            <p className="text-sm text-[#8b98a5]">
                                Context: RANGING → 75% allocation. Effective size: $750 approved.
                            </p>
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    );
}
