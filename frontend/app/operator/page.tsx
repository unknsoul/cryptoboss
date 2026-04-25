'use client';

/**
 * Operator Page - CRYPTOBOSS vFINAL
 * 
 * Purpose: System control panel - fetches from backend
 * Rules:
 * - NO mock data - fetch from /api/operator
 * - Zero incident count for new accounts
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface OperatorState {
    trading_paused: boolean;
    pause_reason?: string;
    last_action?: string;
    last_action_by?: string;
    uptime_seconds: number;
}

interface IncidentState {
    state: string;
    reason?: string;
    since?: string;
    auto_recoverable: boolean;
}

const emptyOperatorState: OperatorState = {
    trading_paused: false,
    uptime_seconds: 0
};

const emptyIncidentState: IncidentState = {
    state: 'normal',
    auto_recoverable: true
};

export default function OperatorPage() {
    const { activeAccount, token } = useAuth();
    const [state, setState] = useState<OperatorState>(emptyOperatorState);
    const [incidentState, setIncidentState] = useState<IncidentState>(emptyIncidentState);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [actionLoading, setActionLoading] = useState(false);

    const fetchState = useCallback(async () => {
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const [opRes, incRes] = await Promise.all([
                fetch(`${API_URL}/api/operator`, { headers: { 'Authorization': `Bearer ${token}` } }),
                fetch(`${API_URL}/api/incident-state`, { headers: { 'Authorization': `Bearer ${token}` } })
            ]);

            if (opRes.ok) {
                const opPayload = await opRes.json();
                setState(unwrapApiData<OperatorState>(opPayload) || emptyOperatorState);
            }
            if (incRes.ok) {
                const incPayload = await incRes.json();
                setIncidentState(unwrapApiData<IncidentState>(incPayload) || emptyIncidentState);
            }
            setError(null);
        } catch (e: any) {
            console.error('Operator fetch error:', e);
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        setState(emptyOperatorState);
        setIncidentState(emptyIncidentState);
        setLoading(true);
        fetchState();
    }, [activeAccount, fetchState]);

    useEffect(() => {
        const interval = setInterval(fetchState, 10000);
        return () => clearInterval(interval);
    }, [fetchState]);

    const handlePause = async () => {
        if (!token) return;
        setActionLoading(true);
        try {
            await fetch(`${API_URL}/api/operator/pause`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason: 'Manual pause from dashboard' })
            });
            fetchState();
        } catch (e: any) {
            setError(e.message);
        } finally {
            setActionLoading(false);
        }
    };

    const handleResume = async () => {
        if (!token) return;
        setActionLoading(true);
        try {
            await fetch(`${API_URL}/api/operator/resume`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason: 'Manual resume from dashboard' })
            });
            fetchState();
        } catch (e: any) {
            setError(e.message);
        } finally {
            setActionLoading(false);
        }
    };

    const formatUptime = (seconds: number) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return `${h}h ${m}m`;
    };

    const getStateColor = (s: string) => {
        switch (s) {
            case 'normal': return 'bg-green-500/20 text-green-400';
            case 'degraded': return 'bg-yellow-500/20 text-yellow-400';
            case 'incident_freeze': return 'bg-orange-500/20 text-orange-400';
            case 'halted': return 'bg-red-500/20 text-red-400';
            default: return 'bg-gray-500/20 text-gray-400';
        }
    };

    return (
        <div className="p-6 space-y-6">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-1">Operator Panel</h1>
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
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Trading Status */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                        <h2 className="text-white font-semibold mb-4">Trading Status</h2>
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-gray-400">Status</span>
                            <span className={`px-3 py-1 rounded font-bold ${state.trading_paused ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                                }`}>
                                {state.trading_paused ? 'PAUSED' : 'ACTIVE'}
                            </span>
                        </div>
                        {state.pause_reason && (
                            <div className="text-sm text-gray-400 mb-4">
                                Reason: {state.pause_reason}
                            </div>
                        )}
                        <div className="flex gap-3">
                            <button
                                onClick={handlePause}
                                disabled={state.trading_paused || actionLoading}
                                className="flex-1 py-2 px-4 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 disabled:opacity-50"
                            >
                                Pause
                            </button>
                            <button
                                onClick={handleResume}
                                disabled={!state.trading_paused || actionLoading}
                                className="flex-1 py-2 px-4 rounded bg-green-500/20 text-green-400 hover:bg-green-500/30 disabled:opacity-50"
                            >
                                Resume
                            </button>
                        </div>
                    </div>

                    {/* Incident State */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                        <h2 className="text-white font-semibold mb-4">Incident State</h2>
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-gray-400">Current State</span>
                            <span className={`px-3 py-1 rounded font-bold uppercase ${getStateColor(incidentState.state)}`}>
                                {incidentState.state.replace('_', ' ')}
                            </span>
                        </div>
                        {incidentState.reason && (
                            <div className="text-sm text-gray-400 mb-2">
                                Reason: {incidentState.reason}
                            </div>
                        )}
                        <div className="text-sm text-gray-500">
                            Auto-recoverable: {incidentState.auto_recoverable ? 'Yes' : 'No'}
                        </div>
                    </div>

                    {/* System Info */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6 lg:col-span-2">
                        <h2 className="text-white font-semibold mb-4">System Info</h2>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                            <div>
                                <div className="text-gray-400 text-sm">Uptime</div>
                                <div className="text-white font-medium">{formatUptime(state.uptime_seconds)}</div>
                            </div>
                            <div>
                                <div className="text-gray-400 text-sm">Last Action</div>
                                <div className="text-white font-medium">{state.last_action || 'None'}</div>
                            </div>
                            <div>
                                <div className="text-gray-400 text-sm">Action By</div>
                                <div className="text-white font-medium">{state.last_action_by || '-'}</div>
                            </div>
                            <div>
                                <div className="text-gray-400 text-sm">Account</div>
                                <div className="text-white font-medium">{activeAccount?.environment || '-'}</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
