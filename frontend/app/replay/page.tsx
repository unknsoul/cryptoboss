'use client';

/**
 * Replay & Analysis Page
 * 
 * Purpose: Post-trade validation and learning
 * Rules:
 * - Replay must mirror live decisions
 * - No simulated profits exaggeration
 */

import { useState } from 'react';

// Mock replay sessions
const replaySessions = [
    {
        id: '2024-01-28',
        date: 'Today',
        trades: 3,
        decisions: 24,
        pnl: 42.50,
        status: 'complete'
    },
    {
        id: '2024-01-27',
        date: 'Yesterday',
        trades: 5,
        decisions: 31,
        pnl: -28.00,
        status: 'complete'
    },
    {
        id: '2024-01-26',
        date: 'Jan 26',
        trades: 4,
        decisions: 28,
        pnl: 85.20,
        status: 'complete'
    },
];

const replayDecisions = [
    {
        time: '14:32:15',
        type: 'TRADE_EXECUTED',
        live: { action: 'BUY 0.025 BTC @ 89168', result: '+$6.21' },
        replay: { action: 'BUY 0.025 BTC @ 89168', result: '+$6.21' },
        match: true,
    },
    {
        time: '14:28:42',
        type: 'CONTEXT_CHANGE',
        live: { action: 'TRENDING_UP → RANGING', result: 'Cooldown applied' },
        replay: { action: 'TRENDING_UP → RANGING', result: 'Cooldown applied' },
        match: true,
    },
    {
        time: '14:15:00',
        type: 'PROPOSAL_REJECTED',
        live: { action: 'Long entry blocked', result: 'Budget exhausted' },
        replay: { action: 'Long entry blocked', result: 'Budget exhausted' },
        match: true,
    },
];

export default function ReplayAnalysisPage() {
    const [selectedSession, setSelectedSession] = useState(replaySessions[0].id);

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Replay & Analysis</h1>
                <p className="text-[#8b98a5] text-sm">
                    Post-trade validation and learning — replay mirrors live decisions
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Session Selector */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Sessions</span>
                    </div>

                    <div className="space-y-2">
                        {replaySessions.map((session) => (
                            <button
                                key={session.id}
                                onClick={() => setSelectedSession(session.id)}
                                className={`w-full p-3 rounded-md text-left transition-colors ${selectedSession === session.id
                                        ? 'bg-[#5b7a9d]/15 border border-[#5b7a9d]'
                                        : 'bg-[#1a1f26] hover:bg-[#242b33]'
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-[#e7e9ea] font-medium">{session.date}</span>
                                    <span className={`text-sm ${session.pnl >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]'
                                        }`}>
                                        {session.pnl >= 0 ? '+' : ''}${session.pnl.toFixed(2)}
                                    </span>
                                </div>
                                <div className="flex items-center gap-4 text-xs text-[#6b7280]">
                                    <span>{session.trades} trades</span>
                                    <span>{session.decisions} decisions</span>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Decision Playback */}
                <div className="card lg:col-span-2">
                    <div className="card-header">
                        <span className="card-title">Decision Comparison</span>
                        <span className="badge badge-success">All Matched</span>
                    </div>

                    <div className="space-y-4">
                        {replayDecisions.map((decision, idx) => (
                            <div
                                key={idx}
                                className={`p-4 rounded-md ${decision.match ? 'bg-[#1a1f26]' : 'bg-[#a65454]/10'
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs font-mono text-[#6b7280]">
                                            {decision.time}
                                        </span>
                                        <span className="badge badge-neutral">{decision.type}</span>
                                    </div>
                                    <span className={`badge ${decision.match ? 'badge-success' : 'badge-danger'}`}>
                                        {decision.match ? '✓ MATCH' : '✗ MISMATCH'}
                                    </span>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <span className="label block mb-1">Live</span>
                                        <div className="text-sm">
                                            <span className="text-[#e7e9ea]">{decision.live.action}</span>
                                            <span className="text-[#8b98a5] ml-2">→ {decision.live.result}</span>
                                        </div>
                                    </div>
                                    <div>
                                        <span className="label block mb-1">Replay</span>
                                        <div className="text-sm">
                                            <span className="text-[#e7e9ea]">{decision.replay.action}</span>
                                            <span className="text-[#8b98a5] ml-2">→ {decision.replay.result}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Mismatch Alert */}
            <div className="card bg-[#1a1f26]">
                <div className="flex items-start gap-4">
                    <span className="text-2xl">💡</span>
                    <div>
                        <h3 className="text-[#e7e9ea] font-medium mb-1">
                            Understanding Replay Validation
                        </h3>
                        <p className="text-sm text-[#8b98a5]">
                            Replay validation ensures that given the same market conditions, the system would
                            make identical decisions. Mismatches may indicate non-deterministic behavior or
                            data inconsistencies that should be investigated. This is a key audit and
                            compliance feature.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
