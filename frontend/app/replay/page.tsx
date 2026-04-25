'use client';

/**
 * Replay & Analysis Page
 * 
 * CRYPTOBOSS 2.0: Replay is DISABLED by default
 * - Replay must NEVER auto-load
 * - Replay data must NEVER be shown in live/testnet mode
 * - Replay only starts via explicit UI toggle
 */

import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ReplaySession {
    id: string;
    date: string;
    trades: number;
    decisions: number;
    pnl: number;
    status: string;
}

interface ReplayDecision {
    time: string;
    type: string;
    live: { action: string; result: string };
    replay: { action: string; result: string };
    match: boolean;
}

export default function ReplayAnalysisPage() {
    const { activeAccount, token } = useAuth();

    // CRITICAL: Replay is DISABLED by default
    const [replayEnabled, setReplayEnabled] = useState(false);
    const [sessions, setSessions] = useState<ReplaySession[]>([]);
    const [decisions, setDecisions] = useState<ReplayDecision[]>([]);
    const [selectedSession, setSelectedSession] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    // Clear replay data when account changes
    useEffect(() => {
        setSessions([]);
        setDecisions([]);
        setSelectedSession(null);
        setReplayEnabled(false);  // Always reset to disabled
    }, [activeAccount?.exchange_account_id]);

    // Only fetch sessions when replay is explicitly enabled
    const enableReplay = async () => {
        if (!activeAccount || !token) return;

        setReplayEnabled(true);
        setLoading(true);

        try {
            const res = await fetch(
                `${API_URL}/api/replay/sessions?exchange_account_id=${activeAccount.exchange_account_id}`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (res.ok) {
                const payload = await res.json();
                const data: any = unwrapApiData(payload);
                setSessions(data.sessions || []);
            }
        } catch (error) {
            console.error('Failed to fetch replay sessions:', error);
        } finally {
            setLoading(false);
        }
    };

    const disableReplay = () => {
        setReplayEnabled(false);
        setSessions([]);
        setDecisions([]);
        setSelectedSession(null);
    };

    // Fetch decisions for selected session
    const loadSession = async (sessionId: string) => {
        if (!token || !activeAccount) return;

        setSelectedSession(sessionId);
        setLoading(true);

        try {
            const res = await fetch(
                `${API_URL}/api/replay/session/${sessionId}?exchange_account_id=${activeAccount.exchange_account_id}`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (res.ok) {
                const payload = await res.json();
                const data: any = unwrapApiData(payload);
                setDecisions(data.decisions || []);
            }
        } catch (error) {
            console.error('Failed to load session:', error);
        } finally {
            setLoading(false);
        }
    };

    // Show disabled state by default
    if (!replayEnabled) {
        return (
            <div className="space-y-6">
                {/* Page Header */}
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Replay & Analysis</h1>
                    <p className="text-[#8b98a5] text-sm">
                        Post-trade validation and learning — replay mirrors live decisions
                    </p>
                </div>

                {/* Disabled State */}
                <div className="card text-center py-12">
                    <div className="text-4xl mb-4">🔒</div>
                    <h2 className="text-xl text-[#e7e9ea] font-medium mb-2">
                        Replay Mode Disabled
                    </h2>
                    <p className="text-[#8b98a5] mb-6 max-w-md mx-auto">
                        Replay mode is disabled by default in live/testnet trading to prevent
                        confusion between historical and live data.
                    </p>

                    {activeAccount ? (
                        <button
                            onClick={enableReplay}
                            className="btn-primary px-6 py-2"
                        >
                            Enable Replay Mode
                        </button>
                    ) : (
                        <p className="text-[#c9a227]">
                            ⚠️ Select an exchange account to enable replay
                        </p>
                    )}
                </div>

                {/* Info Card */}
                <div className="card bg-[#1a1f26]">
                    <div className="flex items-start gap-4">
                        <span className="text-2xl">💡</span>
                        <div>
                            <h3 className="text-[#e7e9ea] font-medium mb-1">
                                Understanding Replay Validation
                            </h3>
                            <p className="text-sm text-[#8b98a5]">
                                Replay validation ensures that given the same market conditions, the system would
                                make identical decisions. This is a key audit and compliance feature.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Show empty state when no sessions
    if (sessions.length === 0 && !loading) {
        return (
            <div className="space-y-6">
                <div className="mb-8 flex items-center justify-between">
                    <div>
                        <h1 className="heading-lg mb-1">Replay & Analysis</h1>
                        <p className="text-[#8b98a5] text-sm">
                            Post-trade validation — <span className="text-[#4a9268]">Replay Mode Active</span>
                        </p>
                    </div>
                    <button
                        onClick={disableReplay}
                        className="text-sm text-[#a65454] hover:text-[#c77777]"
                    >
                        Disable Replay
                    </button>
                </div>

                <div className="card text-center py-12">
                    <div className="text-4xl mb-4">📭</div>
                    <h2 className="text-xl text-[#e7e9ea] font-medium mb-2">
                        No Replay Sessions
                    </h2>
                    <p className="text-[#8b98a5]">
                        No recorded trading sessions found for this account.
                    </p>
                </div>
            </div>
        );
    }

    // Main replay UI (only shown when enabled AND has data)
    return (
        <div className="space-y-6">
            <div className="mb-8 flex items-center justify-between">
                <div>
                    <h1 className="heading-lg mb-1">Replay & Analysis</h1>
                    <p className="text-[#8b98a5] text-sm">
                        Post-trade validation — <span className="text-[#4a9268]">Replay Mode Active</span>
                    </p>
                </div>
                <button
                    onClick={disableReplay}
                    className="text-sm text-[#a65454] hover:text-[#c77777]"
                >
                    Disable Replay
                </button>
            </div>

            {loading ? (
                <div className="card text-center py-12">
                    <div className="animate-spin text-4xl mb-4">⏳</div>
                    <p className="text-[#8b98a5]">Loading replay data...</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Session Selector */}
                    <div className="card">
                        <div className="card-header">
                            <span className="card-title">Sessions</span>
                        </div>

                        <div className="space-y-2">
                            {sessions.map((session) => (
                                <button
                                    key={session.id}
                                    onClick={() => loadSession(session.id)}
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
                            {decisions.length > 0 && decisions.every(d => d.match) && (
                                <span className="badge badge-success">All Matched</span>
                            )}
                        </div>

                        {decisions.length === 0 ? (
                            <div className="text-center py-8 text-[#6b7280]">
                                Select a session to view decisions
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {decisions.map((decision, idx) => (
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
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
