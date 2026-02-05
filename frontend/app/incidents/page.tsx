'use client';

/**
 * Incidents Page - CRYPTOBOSS vFINAL
 * 
 * Purpose: Display incident history from backend
 * Rules:
 * - NO mock data - fetch from /api/incident-state
 * - Zero incidents for new accounts
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface IncidentSnapshot {
    state: string;
    since?: string;
    reason?: string;
    triggered_by?: string;
    auto_recoverable: boolean;
    incident_count_today: number;
    time_in_state_seconds: number;
}

interface TimelineEvent {
    timestamp: string;
    from_state: string;
    to_state: string;
    reason?: string;
    triggered_by?: string;
}

const emptySnapshot: IncidentSnapshot = {
    state: 'normal',
    auto_recoverable: true,
    incident_count_today: 0,
    time_in_state_seconds: 0
};

export default function IncidentsPage() {
    const { activeAccount, token } = useAuth();
    const [snapshot, setSnapshot] = useState<IncidentSnapshot>(emptySnapshot);
    const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/incident-state`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (!response.ok) throw new Error('Failed to fetch incident data');
            const data = await response.json();
            setSnapshot(data.data || emptySnapshot);
            setTimeline(data.data?.timeline || []);
            setError(null);
        } catch (e: any) {
            console.error('Incidents fetch error:', e);
            setError(e.message);
            setSnapshot(emptySnapshot);
            setTimeline([]);
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        setSnapshot(emptySnapshot);
        setTimeline([]);
        setLoading(true);
        fetchData();
    }, [activeAccount, fetchData]);

    useEffect(() => {
        const interval = setInterval(fetchData, 15000);
        return () => clearInterval(interval);
    }, [fetchData]);

    const getStateColor = (state: string) => {
        switch (state) {
            case 'normal': return 'bg-green-500/20 text-green-400 border-green-500/50';
            case 'degraded': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
            case 'incident_freeze': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
            case 'halted': return 'bg-red-500/20 text-red-400 border-red-500/50';
            default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
        }
    };

    const formatDuration = (seconds: number) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (h > 0) return `${h}h ${m}m`;
        return `${m}m`;
    };

    return (
        <div className="p-6 space-y-6">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-1">Incident State</h1>
                <p className="text-gray-400 text-sm">
                    {activeAccount ? `Account: ${activeAccount.label}` : 'No account selected'}
                </p>
            </div>

            {loading && <div className="text-center py-12 text-gray-400">Loading...</div>}
            {error && <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-400 mb-4">Error: {error}</div>}

            {!loading && !activeAccount && (
                <div className="text-center py-12">
                    <div className="text-5xl mb-4">🔐</div>
                    <div className="text-xl text-white mb-2">No Account Selected</div>
                    <div className="text-gray-400">Please select an exchange account</div>
                </div>
            )}

            {!loading && activeAccount && (
                <>
                    {/* Current State */}
                    <div className={`rounded-xl p-6 border ${getStateColor(snapshot.state)}`}>
                        <div className="flex items-center justify-between">
                            <div>
                                <h2 className="text-2xl font-bold uppercase">{snapshot.state.replace('_', ' ')}</h2>
                                {snapshot.reason && <p className="text-sm opacity-80 mt-1">{snapshot.reason}</p>}
                            </div>
                            <div className="text-right">
                                <div className="text-sm opacity-80">Time in state</div>
                                <div className="text-xl font-bold">{formatDuration(snapshot.time_in_state_seconds)}</div>
                            </div>
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <div className="text-gray-400 text-sm">Incidents Today</div>
                            <div className="text-2xl font-bold text-white">{snapshot.incident_count_today}</div>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <div className="text-gray-400 text-sm">Triggered By</div>
                            <div className="text-lg font-medium text-white">{snapshot.triggered_by || 'system'}</div>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <div className="text-gray-400 text-sm">Auto-Recoverable</div>
                            <div className={`text-lg font-bold ${snapshot.auto_recoverable ? 'text-green-400' : 'text-red-400'}`}>
                                {snapshot.auto_recoverable ? 'Yes' : 'No'}
                            </div>
                        </div>
                    </div>

                    {/* Timeline */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                        <h2 className="text-white font-semibold mb-4">State Timeline</h2>
                        {timeline.length === 0 ? (
                            <div className="text-center py-8 text-gray-500">
                                No state changes recorded
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {timeline.map((event, i) => (
                                    <div key={i} className="flex items-center gap-4 text-sm">
                                        <div className="text-gray-500 w-40">
                                            {new Date(event.timestamp).toLocaleString()}
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className={`px-2 py-0.5 rounded text-xs ${getStateColor(event.from_state)}`}>
                                                {event.from_state}
                                            </span>
                                            <span className="text-gray-500">→</span>
                                            <span className={`px-2 py-0.5 rounded text-xs ${getStateColor(event.to_state)}`}>
                                                {event.to_state}
                                            </span>
                                        </div>
                                        <div className="text-gray-400 flex-1">{event.reason}</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
