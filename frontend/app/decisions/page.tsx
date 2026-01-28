'use client';

/**
 * Decision Flow Page
 * 
 * Purpose: Explain why the system acted or did nothing
 * Rules:
 * - Readable text explanations
 * - Chronological order
 * - No raw logs by default
 */

// Mock decision data
const decisions = [
    {
        id: 1,
        timestamp: '14:32:15',
        type: 'PROPOSAL_REJECTED',
        summary: 'Long entry proposal rejected',
        details: [
            { label: 'Reason', value: 'Trade budget exhausted for current context' },
            { label: 'Context', value: 'RANGING' },
            { label: 'Trades Used', value: '3/3' },
        ],
        variant: 'warning' as const,
    },
    {
        id: 2,
        timestamp: '14:28:42',
        type: 'CONTEXT_CHANGE',
        summary: 'Market context changed to RANGING',
        details: [
            { label: 'Previous', value: 'TRENDING_UP' },
            { label: 'Confidence', value: '78%' },
            { label: 'Cooldown', value: '15 minutes applied' },
        ],
        variant: 'neutral' as const,
    },
    {
        id: 3,
        timestamp: '14:15:00',
        type: 'TRADE_EXECUTED',
        summary: 'Short exit executed successfully',
        details: [
            { label: 'Symbol', value: 'BTC/USDT' },
            { label: 'P&L', value: '+$42.50 (0.85%)' },
            { label: 'Reason', value: 'Target profit reached' },
        ],
        variant: 'success' as const,
    },
    {
        id: 4,
        timestamp: '13:45:22',
        type: 'PROPOSAL_REJECTED',
        summary: 'Entry blocked by risk guardian',
        details: [
            { label: 'Reason', value: 'Daily loss limit approaching (85% used)' },
            { label: 'Limit', value: '-$425 / -$500' },
            { label: 'Action', value: 'Trade suspended until reset' },
        ],
        variant: 'danger' as const,
    },
    {
        id: 5,
        timestamp: '13:30:00',
        type: 'NO_ACTION',
        summary: 'No trade signal generated',
        details: [
            { label: 'Reason', value: 'Bias conviction below threshold' },
            { label: 'Required', value: '65%' },
            { label: 'Current', value: '52%' },
        ],
        variant: 'neutral' as const,
    },
    {
        id: 6,
        timestamp: '12:55:18',
        type: 'BIAS_CHANGE',
        summary: 'Bias shifted to LONG_BIAS',
        details: [
            { label: 'Previous', value: 'NEUTRAL' },
            { label: 'Conviction', value: '68%' },
            { label: 'Signal', value: 'Higher lows on 4H' },
        ],
        variant: 'neutral' as const,
    },
];

const decisionTypeLabels: Record<string, { text: string; icon: string }> = {
    'TRADE_EXECUTED': { text: 'Trade', icon: '✓' },
    'PROPOSAL_REJECTED': { text: 'Rejected', icon: '✗' },
    'CONTEXT_CHANGE': { text: 'Context', icon: '↻' },
    'BIAS_CHANGE': { text: 'Bias', icon: '↔' },
    'NO_ACTION': { text: 'No Action', icon: '—' },
};

function DecisionCard({ decision }: { decision: typeof decisions[0] }) {
    const typeInfo = decisionTypeLabels[decision.type] || { text: decision.type, icon: '•' };

    const variantClasses = {
        success: 'border-l-[#4a9268]',
        warning: 'border-l-[#c4a052]',
        danger: 'border-l-[#a65454]',
        neutral: 'border-l-[#6b7280]',
    };

    const badgeClasses = {
        success: 'badge-success',
        warning: 'badge-warning',
        danger: 'badge-danger',
        neutral: 'badge-neutral',
    };

    return (
        <div className={`card border-l-4 ${variantClasses[decision.variant]}`}>
            <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-[#6b7280]">{decision.timestamp}</span>
                    <span className={`badge ${badgeClasses[decision.variant]}`}>
                        {typeInfo.icon} {typeInfo.text}
                    </span>
                </div>
            </div>

            <h3 className="text-[#e7e9ea] font-medium mb-3">{decision.summary}</h3>

            <div className="space-y-2">
                {decision.details.map((detail, idx) => (
                    <div key={idx} className="flex items-center justify-between text-sm">
                        <span className="text-[#6b7280]">{detail.label}</span>
                        <span className="text-[#8b98a5]">{detail.value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default function DecisionFlowPage() {
    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Decision Flow</h1>
                <p className="text-[#8b98a5] text-sm">
                    Explain why the system acted or did nothing — every decision is traceable
                </p>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="card text-center">
                    <span className="label">Today's Decisions</span>
                    <span className="value-lg block mt-1">24</span>
                </div>
                <div className="card text-center">
                    <span className="label">Trades Executed</span>
                    <span className="value-lg block mt-1 text-[#4a9268]">3</span>
                </div>
                <div className="card text-center">
                    <span className="label">Proposals Rejected</span>
                    <span className="value-lg block mt-1 text-[#c4a052]">8</span>
                </div>
                <div className="card text-center">
                    <span className="label">No Action (Normal)</span>
                    <span className="value-lg block mt-1 text-[#6b7280]">13</span>
                </div>
            </div>

            {/* Filter bar */}
            <div className="flex items-center gap-2 mb-6">
                <button className="btn btn-ghost text-sm">All</button>
                <button className="btn btn-ghost text-sm">Trades</button>
                <button className="btn btn-ghost text-sm">Rejected</button>
                <button className="btn btn-ghost text-sm">Context</button>
                <button className="btn btn-ghost text-sm">Bias</button>
            </div>

            {/* Decision Timeline */}
            <div className="space-y-4">
                {decisions.map((decision) => (
                    <DecisionCard key={decision.id} decision={decision} />
                ))}
            </div>

            {/* Load More */}
            <div className="text-center mt-8">
                <button className="btn btn-ghost">
                    Load Earlier Decisions
                </button>
            </div>

            {/* Explainer Banner */}
            <div className="card mt-8 bg-[#1a1f26]">
                <div className="flex items-start gap-4">
                    <div className="text-2xl">💡</div>
                    <div>
                        <h3 className="text-[#e7e9ea] font-medium mb-1">
                            Understanding "No Action" Decisions
                        </h3>
                        <p className="text-sm text-[#8b98a5]">
                            Most decisions result in no trade — this is normal and expected.
                            The system only acts when all conditions align: context allows trading,
                            bias has sufficient conviction, proposals pass risk checks, and budget is available.
                            A high "No Action" count indicates disciplined risk management.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
