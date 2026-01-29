'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/shared/Card';
import { Badge } from '@/components/shared/Badge';
import { PageHeader } from '@/components/shared/PageHeader';
import { StatusDot } from '@/components/shared/StatusDot';

// Mock data - will be replaced with API calls
const mockState = {
    is_paused: false,
    paused_at: null,
    paused_by: null,
    pause_reason: null,
    requires_manual_recovery: false,
    recovery_required_reason: null
};

const mockIncidentState = {
    state: 'normal',
    since: new Date().toISOString(),
    time_in_state_seconds: 3600
};

const mockActionLog = [
    {
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        action: 'manual_resume',
        operator_id: 'admin',
        reason: 'System verified healthy after maintenance',
        success: true
    },
    {
        timestamp: new Date(Date.now() - 90000000).toISOString(),
        action: 'manual_pause',
        operator_id: 'admin',
        reason: 'Scheduled maintenance window',
        success: true
    }
];

type OperatorState = typeof mockState;
type IncidentState = typeof mockIncidentState;
type ActionLog = typeof mockActionLog[0];

export default function OperatorPage() {
    const [state, setState] = useState<OperatorState>(mockState);
    const [incidentState, setIncidentState] = useState<IncidentState>(mockIncidentState);
    const [actionLog, setActionLog] = useState<ActionLog[]>(mockActionLog);
    const [reason, setReason] = useState('');
    const [isConfirmOpen, setIsConfirmOpen] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handlePause = async () => {
        if (!reason.trim()) return;
        setLoading(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setState(prev => ({
            ...prev,
            is_paused: true,
            paused_at: new Date().toISOString(),
            paused_by: 'operator',
            pause_reason: reason
        }));
        setActionLog(prev => [{
            timestamp: new Date().toISOString(),
            action: 'manual_pause',
            operator_id: 'operator',
            reason: reason,
            success: true
        }, ...prev]);
        setReason('');
        setIsConfirmOpen(null);
        setLoading(false);
    };

    const handleResume = async () => {
        if (!reason.trim()) return;
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 1000));
        setState(prev => ({
            ...prev,
            is_paused: false,
            paused_at: null,
            paused_by: null,
            pause_reason: null
        }));
        setActionLog(prev => [{
            timestamp: new Date().toISOString(),
            action: 'manual_resume',
            operator_id: 'operator',
            reason: reason,
            success: true
        }, ...prev]);
        setReason('');
        setIsConfirmOpen(null);
        setLoading(false);
    };

    const handleRecover = async () => {
        if (!reason.trim()) return;
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 1500));
        setState(prev => ({
            ...prev,
            is_paused: false,
            requires_manual_recovery: false,
            recovery_required_reason: null
        }));
        setIncidentState(prev => ({
            ...prev,
            state: 'normal',
            since: new Date().toISOString()
        }));
        setActionLog(prev => [{
            timestamp: new Date().toISOString(),
            action: 'manual_recover',
            operator_id: 'operator',
            reason: reason,
            success: true
        }, ...prev]);
        setReason('');
        setIsConfirmOpen(null);
        setLoading(false);
    };

    const getStateColor = (s: string): 'success' | 'warning' | 'danger' | 'neutral' => {
        switch (s) {
            case 'normal': return 'success';
            case 'degraded': return 'warning';
            case 'incident_freeze': return 'danger';
            case 'halted': return 'danger';
            default: return 'neutral';
        }
    };

    const formatTime = (seconds: number): string => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Operator Control"
                description="Manual system control - pause, resume, and recover from incidents"
            />

            {/* System State Overview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card title="System Status">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <StatusDot
                                status={state.is_paused ? 'warning' : 'success'}
                                pulse={!state.is_paused}
                            />
                            <span className="text-lg font-medium text-[#e7e9ea]">
                                {state.is_paused ? 'PAUSED' : 'RUNNING'}
                            </span>
                        </div>
                        {state.paused_by && (
                            <span className="text-sm text-[#8b98a5]">
                                by {state.paused_by}
                            </span>
                        )}
                    </div>
                    {state.pause_reason && (
                        <p className="mt-3 text-sm text-[#8b98a5]">
                            Reason: {state.pause_reason}
                        </p>
                    )}
                </Card>

                <Card title="Incident State">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <StatusDot
                                status={getStateColor(incidentState.state)}
                                pulse={incidentState.state !== 'normal'}
                            />
                            <span className="text-lg font-medium text-[#e7e9ea] uppercase">
                                {incidentState.state.replace('_', ' ')}
                            </span>
                        </div>
                    </div>
                    <p className="mt-3 text-sm text-[#8b98a5]">
                        Time in state: {formatTime(incidentState.time_in_state_seconds)}
                    </p>
                </Card>

                <Card title="Recovery Required">
                    <div className="flex items-center justify-between">
                        <Badge
                            variant={state.requires_manual_recovery ? 'danger' : 'success'}
                        >
                            {state.requires_manual_recovery ? 'YES' : 'NO'}
                        </Badge>
                    </div>
                    {state.recovery_required_reason && (
                        <p className="mt-3 text-sm text-[#e06c75]">
                            {state.recovery_required_reason}
                        </p>
                    )}
                </Card>
            </div>

            {/* Control Actions */}
            <Card title="Control Actions">
                <div className="space-y-4">
                    <p className="text-sm text-[#8b98a5]">
                        All operator actions are logged and require a reason.
                        <span className="text-[#e06c75]"> These actions cannot override risk or capital vetoes.</span>
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Pause Button */}
                        <div className={`p-4 rounded-lg border ${state.is_paused ? 'bg-[#1a1f26] border-[#2d3640] opacity-50' : 'bg-[#141920] border-[#2d3640] hover:border-[#c9a227]'} transition-all`}>
                            <h4 className="font-medium text-[#e7e9ea] mb-2">Pause Trading</h4>
                            <p className="text-sm text-[#8b98a5] mb-3">
                                Immediately pause all trading activity.
                            </p>
                            <button
                                onClick={() => setIsConfirmOpen('pause')}
                                disabled={state.is_paused}
                                className="w-full py-2 px-4 bg-[#c9a227] text-black font-medium rounded-md hover:bg-[#d4b13c] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                Pause System
                            </button>
                        </div>

                        {/* Resume Button */}
                        <div className={`p-4 rounded-lg border ${!state.is_paused || state.requires_manual_recovery ? 'bg-[#1a1f26] border-[#2d3640] opacity-50' : 'bg-[#141920] border-[#2d3640] hover:border-[#4a9268]'} transition-all`}>
                            <h4 className="font-medium text-[#e7e9ea] mb-2">Resume Trading</h4>
                            <p className="text-sm text-[#8b98a5] mb-3">
                                Resume after health validation passes.
                            </p>
                            <button
                                onClick={() => setIsConfirmOpen('resume')}
                                disabled={!state.is_paused || state.requires_manual_recovery}
                                className="w-full py-2 px-4 bg-[#4a9268] text-white font-medium rounded-md hover:bg-[#5aa878] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                Resume System
                            </button>
                        </div>

                        {/* Recover Button */}
                        <div className={`p-4 rounded-lg border ${!state.requires_manual_recovery ? 'bg-[#1a1f26] border-[#2d3640] opacity-50' : 'bg-[#141920] border-[#e06c75]/20 hover:border-[#e06c75]'} transition-all`}>
                            <h4 className="font-medium text-[#e7e9ea] mb-2">Recovery from Halt</h4>
                            <p className="text-sm text-[#8b98a5] mb-3">
                                Recover after critical incident resolution.
                            </p>
                            <button
                                onClick={() => setIsConfirmOpen('recover')}
                                disabled={!state.requires_manual_recovery}
                                className="w-full py-2 px-4 bg-[#e06c75] text-white font-medium rounded-md hover:bg-[#e58089] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                Recover System
                            </button>
                        </div>
                    </div>
                </div>
            </Card>

            {/* Action Log */}
            <Card title="Operator Action Log" subtitle="Last 24 hours">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-[#2d3640]">
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Time</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Action</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Operator</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Reason</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {actionLog.length === 0 ? (
                                <tr>
                                    <td colSpan={5} className="py-8 text-center text-[#8b98a5]">
                                        No actions recorded in the last 24 hours
                                    </td>
                                </tr>
                            ) : (
                                actionLog.map((log, i) => (
                                    <tr key={i} className="border-b border-[#2d3640]/50 hover:bg-[#1a1f26] transition-colors">
                                        <td className="py-3 px-4 text-sm text-[#8b98a5]">
                                            {new Date(log.timestamp).toLocaleString()}
                                        </td>
                                        <td className="py-3 px-4">
                                            <Badge variant={
                                                log.action === 'manual_pause' ? 'warning' :
                                                    log.action === 'manual_resume' ? 'success' :
                                                        log.action === 'manual_recover' ? 'info' : 'neutral'
                                            }>
                                                {log.action.replace('_', ' ').toUpperCase()}
                                            </Badge>
                                        </td>
                                        <td className="py-3 px-4 text-sm text-[#e7e9ea] font-mono">
                                            {log.operator_id}
                                        </td>
                                        <td className="py-3 px-4 text-sm text-[#8b98a5] max-w-xs truncate">
                                            {log.reason}
                                        </td>
                                        <td className="py-3 px-4">
                                            <StatusDot status={log.success ? 'success' : 'danger'} />
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </Card>

            {/* Confirmation Modal */}
            {isConfirmOpen && (
                <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
                    <div className="bg-[#141920] border border-[#2d3640] rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
                        <h3 className="text-lg font-semibold text-[#e7e9ea] mb-2">
                            {isConfirmOpen === 'pause' && 'Confirm System Pause'}
                            {isConfirmOpen === 'resume' && 'Confirm System Resume'}
                            {isConfirmOpen === 'recover' && 'Confirm System Recovery'}
                        </h3>
                        <p className="text-sm text-[#8b98a5] mb-4">
                            {isConfirmOpen === 'pause' && 'This will immediately stop all trading activity.'}
                            {isConfirmOpen === 'resume' && 'System health will be validated before resuming.'}
                            {isConfirmOpen === 'recover' && 'This will attempt to recover from a critical halt. Ensure the issue is fully resolved.'}
                        </p>

                        <div className="mb-4">
                            <label className="block text-sm font-medium text-[#8b98a5] mb-2">
                                Reason (required)
                            </label>
                            <textarea
                                value={reason}
                                onChange={(e) => setReason(e.target.value)}
                                placeholder="Enter the reason for this action..."
                                className="w-full px-3 py-2 bg-[#0f1419] border border-[#2d3640] rounded-md text-[#e7e9ea] placeholder-[#6b7280] focus:outline-none focus:border-[#5b7a9d] resize-none"
                                rows={3}
                            />
                        </div>

                        <div className="flex gap-3">
                            <button
                                onClick={() => {
                                    setIsConfirmOpen(null);
                                    setReason('');
                                }}
                                className="flex-1 py-2 px-4 border border-[#2d3640] text-[#8b98a5] rounded-md hover:bg-[#1a1f26] transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    if (isConfirmOpen === 'pause') handlePause();
                                    if (isConfirmOpen === 'resume') handleResume();
                                    if (isConfirmOpen === 'recover') handleRecover();
                                }}
                                disabled={!reason.trim() || loading}
                                className={`flex-1 py-2 px-4 font-medium rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${isConfirmOpen === 'pause' ? 'bg-[#c9a227] text-black hover:bg-[#d4b13c]' :
                                        isConfirmOpen === 'resume' ? 'bg-[#4a9268] text-white hover:bg-[#5aa878]' :
                                            'bg-[#e06c75] text-white hover:bg-[#e58089]'
                                    }`}
                            >
                                {loading ? 'Processing...' : 'Confirm'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
