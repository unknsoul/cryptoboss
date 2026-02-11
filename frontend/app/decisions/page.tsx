'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';

/**
 * Decision Flow Page
 * 
 * Purpose: Explain why the system acted or did nothing
 * Rules:
 * - Readable text explanations
 * - Chronological order
 * - No raw logs by default
 * - NO MOCK DATA - fetch from backend
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Decision {
    id: string;
    timestamp: string;
    type: string;
    summary: string;
    details: Array<{ label: string; value: string }>;
    variant: 'success' | 'warning' | 'danger' | 'neutral';
}

const decisionTypeLabels: Record<string, { text: string; icon: string }> = {
    'TRADE_EXECUTED': { text: 'Trade', icon: '✓' },
    'PROPOSAL_REJECTED': { text: 'Rejected', icon: '✗' },
    'CONTEXT_CHANGE': { text: 'Context', icon: '↻' },
    'BIAS_CHANGE': { text: 'Bias', icon: '↔' },
    'NO_ACTION': { text: 'No Action', icon: '—' },
    'ENTRY': { text: 'Entry', icon: '▶' },
    'EXIT': { text: 'Exit', icon: '■' },
};

function getVariantFromDecision(decision: any): 'success' | 'warning' | 'danger' | 'neutral' {
    if (decision.status === 'executed' || decision.type === 'TRADE_EXECUTED') return 'success';
    if (decision.status === 'rejected') return 'warning';
    if (decision.status === 'cancelled' || decision.type === 'PROPOSAL_REJECTED') return 'danger';
    return 'neutral';
}

function formatDecisionForDisplay(raw: any): Decision {
    const timestamp = new Date(raw.timestamp || raw.created_at || Date.now())
        .toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    return {
        id: raw.decision_id || raw.id || String(Date.now()),
        timestamp,
        type: raw.decision_type || raw.type || 'NO_ACTION',
        summary: raw.narrative || raw.summary || raw.reason || 'Decision recorded',
        details: [
            ...(raw.symbol ? [{ label: 'Symbol', value: raw.symbol }] : []),
            ...(raw.side ? [{ label: 'Side', value: raw.side }] : []),
            ...(raw.context ? [{ label: 'Context', value: raw.context }] : []),
            ...(raw.reason ? [{ label: 'Reason', value: raw.reason }] : []),
        ],
        variant: getVariantFromDecision(raw)
    };
}


function DecisionCard({ decision }: { decision: Decision }) {
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
    const { activeAccount, token } = useAuth();
    const [decisions, setDecisions] = useState<Decision[]>([]);
    const [stats, setStats] = useState({ total: 0, executed: 0, rejected: 0, noAction: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchDecisions = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/v11/decisions?limit=50`, {
                headers: token ? { 'Authorization': `Bearer ${token}` } : {}
            });

            if (!response.ok) {
                throw new Error('Failed to fetch decisions');
            }

            const data = await response.json();
            const formatted = (Array.isArray(data) ? data : []).map(formatDecisionForDisplay);

            setDecisions(formatted);
            setStats({
                total: formatted.length,
                executed: formatted.filter(d => d.type === 'TRADE_EXECUTED' || d.variant === 'success').length,
                rejected: formatted.filter(d => d.type === 'PROPOSAL_REJECTED' || d.variant === 'danger').length,
                noAction: formatted.filter(d => d.type === 'NO_ACTION').length,
            });
            setError(null);
        } catch (e: any) {
            console.error('Decisions fetch error:', e);
            setError(e.message);
            // Empty state on error - no mock data
            setDecisions([]);
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        setLoading(true);
        fetchDecisions();
    }, [activeAccount, fetchDecisions]);

    // Refresh every 10 seconds
    useEffect(() => {
        const interval = setInterval(fetchDecisions, 10000);
        return () => clearInterval(interval);
    }, [fetchDecisions]);

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Decision Flow</h1>
                <p className="text-[#8b98a5] text-sm">
                    Explain why the system acted or did nothing — every decision is traceable
                </p>
            </div>

            {/* Loading State */}
            {loading && (
                <div className="text-center py-12 text-[#8b98a5]">Loading decisions...</div>
            )}

            {/* Error State */}
            {error && (
                <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-400">
                    Error: {error}
                </div>
            )}

            {/* Summary Stats */}
            {!loading && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div className="card text-center">
                        <span className="label">Today's Decisions</span>
                        <span className="value-lg block mt-1">{stats.total}</span>
                    </div>
                    <div className="card text-center">
                        <span className="label">Trades Executed</span>
                        <span className="value-lg block mt-1 text-[#4a9268]">{stats.executed}</span>
                    </div>
                    <div className="card text-center">
                        <span className="label">Proposals Rejected</span>
                        <span className="value-lg block mt-1 text-[#c4a052]">{stats.rejected}</span>
                    </div>
                    <div className="card text-center">
                        <span className="label">No Action (Normal)</span>
                        <span className="value-lg block mt-1 text-[#6b7280]">{stats.noAction}</span>
                    </div>
                </div>
            )}

            {/* Filter bar */}
            <div className="flex items-center gap-2 mb-6">
                <button className="btn btn-ghost text-sm">All</button>
                <button className="btn btn-ghost text-sm">Trades</button>
                <button className="btn btn-ghost text-sm">Rejected</button>
                <button className="btn btn-ghost text-sm">Context</button>
                <button className="btn btn-ghost text-sm">Bias</button>
            </div>

            {/* Decision Timeline */}
            {!loading && decisions.length > 0 && (
                <div className="space-y-4">
                    {decisions.map((decision) => (
                        <DecisionCard key={decision.id} decision={decision} />
                    ))}
                </div>
            )}

            {/* Empty State */}
            {!loading && decisions.length === 0 && (
                <div className="text-center py-12 bg-[#1d2229] border border-[#2d3640] rounded-xl">
                    <div className="text-5xl mb-4">📋</div>
                    <div className="text-xl text-white mb-2">No Decisions Yet</div>
                    <div className="text-gray-400">
                        Decision records will appear here once the system processes market data.
                    </div>
                </div>
            )}

            {/* Load More */}
            {decisions.length > 0 && (
                <div className="text-center mt-8">
                    <button className="btn btn-ghost">
                        Load Earlier Decisions
                    </button>
                </div>
            )}

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
