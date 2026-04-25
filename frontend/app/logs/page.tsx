'use client';

import { useEffect, useCallback, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SystemSnapshot {
    engine_status?: string;
    connection_status?: string;
    incident_state?: string;
    trading_paused?: boolean;
    kill_switch?: {
        active?: boolean;
        reason?: string | null;
    };
}

interface OperatorAction {
    action: string;
    reason?: string;
    timestamp: string;
    operator?: string;
}

interface DecisionItem {
    timestamp?: string;
    symbol?: string;
    action?: string;
    strategy?: string;
    reason?: string;
    outcome?: string;
}

const emptySystem: SystemSnapshot = {
    engine_status: 'stopped',
    connection_status: 'disconnected',
    incident_state: 'NORMAL',
    trading_paused: false,
    kill_switch: {
        active: false,
        reason: null,
    },
};

export default function LogsAuditPage() {
    const { token } = useAuth();
    const [system, setSystem] = useState<SystemSnapshot>(emptySystem);
    const [actions, setActions] = useState<OperatorAction[]>([]);
    const [decisions, setDecisions] = useState<DecisionItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchLogs = useCallback(async () => {
        try {
            const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
            const [systemRes, actionsRes, decisionsRes] = await Promise.all([
                fetch(`${API_URL}/api/system`, { headers, cache: 'no-store' }),
                fetch(`${API_URL}/api/operator/actions?limit=50`, { headers, cache: 'no-store' }),
                fetch(`${API_URL}/api/v11/decisions?limit=25`, { headers, cache: 'no-store' }),
            ]);

            if (!systemRes.ok || !actionsRes.ok || !decisionsRes.ok) {
                throw new Error('Failed to load audit data');
            }

            const [systemPayload, actionsPayload, decisionsPayload] = await Promise.all([
                systemRes.json(),
                actionsRes.json(),
                decisionsRes.json(),
            ]);

            setSystem(unwrapApiData<SystemSnapshot>(systemPayload) || emptySystem);
            setActions(unwrapApiData<any>(actionsPayload)?.actions || []);
            setDecisions(unwrapApiData<any>(decisionsPayload)?.decisions || []);
            setError(null);
        } catch (fetchError) {
            console.error('Failed to load logs:', fetchError);
            setError(fetchError instanceof Error ? fetchError.message : 'Failed to load audit data');
            setSystem(emptySystem);
            setActions([]);
            setDecisions([]);
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        fetchLogs();
    }, [fetchLogs]);

    useEffect(() => {
        const interval = setInterval(fetchLogs, 10000);
        return () => clearInterval(interval);
    }, [fetchLogs]);

    const engineBadgeClass =
        system.engine_status === 'running'
            ? 'badge-success'
            : system.engine_status === 'paused'
                ? 'badge-warning'
                : 'badge-neutral';

    const connectionBadgeClass =
        system.connection_status === 'connected'
            ? 'badge-success'
            : system.connection_status === 'connecting'
                ? 'badge-warning'
                : 'badge-danger';

    return (
        <div className="space-y-6">
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Logs & Audit</h1>
                <p className="text-[#8b98a5] text-sm">
                    Runtime status, operator actions, and recent decision history
                </p>
            </div>

            {loading && (
                <div className="card text-sm text-[#8b98a5]">Loading audit trail...</div>
            )}

            {error && (
                <div className="card text-sm text-[#d28383] border border-[#a65454]">
                    {error}
                </div>
            )}

            {!loading && !error && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Engine</span>
                            </div>
                            <span className={`badge ${engineBadgeClass}`}>
                                {(system.engine_status || 'stopped').toUpperCase()}
                            </span>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Connection</span>
                            </div>
                            <span className={`badge ${connectionBadgeClass}`}>
                                {(system.connection_status || 'disconnected').toUpperCase()}
                            </span>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Incident</span>
                            </div>
                            <span className="badge badge-neutral">
                                {(system.incident_state || 'NORMAL').replace('_', ' ')}
                            </span>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Kill Switch</span>
                            </div>
                            <span className={`badge ${system.kill_switch?.active ? 'badge-danger' : 'badge-success'}`}>
                                {system.kill_switch?.active ? 'ACTIVE' : 'OFF'}
                            </span>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Operator Actions</span>
                                <span className="badge badge-neutral">{actions.length}</span>
                            </div>

                            <div className="space-y-3 max-h-[28rem] overflow-auto pr-1">
                                {actions.length === 0 && (
                                    <div className="text-sm text-[#6b7280]">No operator actions recorded.</div>
                                )}
                                {actions.slice().reverse().map((action, index) => (
                                    <div key={`${action.timestamp}-${index}`} className="rounded-md border border-[#2d3640] p-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <span className="text-[#e7e9ea] font-medium">{action.action}</span>
                                            <span className="text-xs text-[#6b7280]">
                                                {new Date(action.timestamp).toLocaleString()}
                                            </span>
                                        </div>
                                        <div className="text-sm text-[#8b98a5] mt-2">
                                            {action.reason || 'No reason provided'}
                                        </div>
                                        {action.operator && (
                                            <div className="text-xs text-[#6b7280] mt-2">
                                                Operator: {action.operator}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Recent Decisions</span>
                                <span className="badge badge-neutral">{decisions.length}</span>
                            </div>

                            <div className="space-y-3 max-h-[28rem] overflow-auto pr-1">
                                {decisions.length === 0 && (
                                    <div className="text-sm text-[#6b7280]">No recent decisions available.</div>
                                )}
                                {decisions.slice().reverse().map((decision, index) => (
                                    <div key={`${decision.timestamp || 'decision'}-${index}`} className="rounded-md border border-[#2d3640] p-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <span className="text-[#e7e9ea] font-medium">
                                                {decision.action || 'UNKNOWN'} {decision.symbol ? `- ${decision.symbol}` : ''}
                                            </span>
                                            <span className="text-xs text-[#6b7280]">
                                                {decision.timestamp ? new Date(decision.timestamp).toLocaleString() : '--'}
                                            </span>
                                        </div>
                                        <div className="text-sm text-[#8b98a5] mt-2">
                                            {decision.reason || 'No decision note recorded'}
                                        </div>
                                        <div className="flex items-center gap-3 mt-2 text-xs text-[#6b7280]">
                                            <span>Strategy: {decision.strategy || '--'}</span>
                                            <span>Outcome: {decision.outcome || '--'}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
