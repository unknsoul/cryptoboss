'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/shared/Card';
import { Badge } from '@/components/shared/Badge';
import { PageHeader } from '@/components/shared/PageHeader';
import { StatusDot } from '@/components/shared/StatusDot';

// Mock data
const mockIncidentState = {
    state: 'normal',
    since: new Date(Date.now() - 3600000).toISOString(),
    reason: 'System initialized',
    triggered_by: 'system',
    auto_recoverable: true,
    incident_count_today: 1,
    time_in_state_seconds: 3600
};

const mockTimeline = [
    {
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        from_state: 'degraded',
        to_state: 'normal',
        reason: 'Auto-recovery: conditions improved',
        triggered_by: 'system',
        auto_recoverable: true
    },
    {
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        from_state: 'normal',
        to_state: 'degraded',
        reason: 'Exchange latency exceeded warning threshold',
        triggered_by: 'system',
        auto_recoverable: true
    },
    {
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        from_state: 'incident_freeze',
        to_state: 'normal',
        reason: 'Resolved by admin: API connectivity restored',
        triggered_by: 'admin',
        auto_recoverable: false
    },
    {
        timestamp: new Date(Date.now() - 90000000).toISOString(),
        from_state: 'normal',
        to_state: 'incident_freeze',
        reason: 'Exchange API timeout threshold exceeded',
        triggered_by: 'system',
        auto_recoverable: false
    }
];

type IncidentSnapshot = typeof mockIncidentState;
type TimelineEvent = typeof mockTimeline[0];

export default function IncidentsPage() {
    const [snapshot, setSnapshot] = useState<IncidentSnapshot>(mockIncidentState);
    const [timeline, setTimeline] = useState<TimelineEvent[]>(mockTimeline);
    const [filter, setFilter] = useState<string>('all');
    const [loading, setLoading] = useState(false);

    const getStateColor = (state: string): 'success' | 'warning' | 'danger' | 'neutral' => {
        switch (state) {
            case 'normal': return 'success';
            case 'degraded': return 'warning';
            case 'incident_freeze': return 'danger';
            case 'halted': return 'danger';
            default: return 'neutral';
        }
    };

    const getStateBadgeVariant = (state: string): 'success' | 'warning' | 'danger' | 'neutral' => {
        return getStateColor(state);
    };

    const formatDuration = (seconds: number): string => {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    };

    const filteredTimeline = filter === 'all'
        ? timeline
        : timeline.filter(e => e.to_state === filter);

    const stateDescriptions: Record<string, string> = {
        normal: 'Full trading capability. All systems operational.',
        degraded: 'Reduced trading. Position sizes may be limited.',
        incident_freeze: 'No new trades allowed. Existing positions can be managed.',
        halted: 'Complete system stop. Requires manual recovery.'
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Incident Timeline"
                description="System state transitions and incident history"
            />

            {/* Current State */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="md:col-span-2">
                    <div className="flex items-center gap-4">
                        <div className={`w-16 h-16 rounded-full flex items-center justify-center ${snapshot.state === 'normal' ? 'bg-[#4a9268]/20' :
                                snapshot.state === 'degraded' ? 'bg-[#c9a227]/20' :
                                    'bg-[#e06c75]/20'
                            }`}>
                            <StatusDot
                                status={getStateColor(snapshot.state)}
                                pulse={snapshot.state !== 'normal'}
                                size="lg"
                            />
                        </div>
                        <div className="flex-1">
                            <h3 className="text-xl font-semibold text-[#e7e9ea] uppercase">
                                {snapshot.state.replace('_', ' ')}
                            </h3>
                            <p className="text-sm text-[#8b98a5] mt-1">
                                {stateDescriptions[snapshot.state]}
                            </p>
                        </div>
                    </div>
                </Card>

                <Card title="Time in State">
                    <div className="text-2xl font-bold text-[#e7e9ea] font-mono">
                        {formatDuration(snapshot.time_in_state_seconds)}
                    </div>
                    <p className="text-sm text-[#8b98a5] mt-1">
                        Since {new Date(snapshot.since).toLocaleString()}
                    </p>
                </Card>

                <Card title="Incidents Today">
                    <div className="text-2xl font-bold text-[#e7e9ea]">
                        {snapshot.incident_count_today}
                    </div>
                    <p className="text-sm text-[#8b98a5] mt-1">
                        {snapshot.auto_recoverable ? 'Auto-recoverable' : 'Requires operator'}
                    </p>
                </Card>
            </div>

            {/* State Transition Rules */}
            <Card title="State Transition Rules">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {['normal', 'degraded', 'incident_freeze', 'halted'].map(state => (
                        <div
                            key={state}
                            className={`p-4 rounded-lg border ${snapshot.state === state
                                    ? 'border-[#5b7a9d] bg-[#5b7a9d]/10'
                                    : 'border-[#2d3640] bg-[#0f1419]'
                                }`}
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <StatusDot status={getStateColor(state)} />
                                <span className="text-sm font-medium text-[#e7e9ea] uppercase">
                                    {state.replace('_', ' ')}
                                </span>
                            </div>
                            <div className="space-y-1 text-xs text-[#8b98a5]">
                                <div className="flex items-center gap-2">
                                    <span className={state === 'halted' || state === 'incident_freeze' ? 'text-[#e06c75]' : 'text-[#4a9268]'}>
                                        {state === 'halted' || state === 'incident_freeze' ? '✗' : '✓'}
                                    </span>
                                    New Trades
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={state === 'halted' ? 'text-[#e06c75]' : 'text-[#4a9268]'}>
                                        {state === 'halted' ? '✗' : '✓'}
                                    </span>
                                    Manage Positions
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={state === 'incident_freeze' || state === 'halted' ? 'text-[#e06c75]' : state === 'degraded' ? 'text-[#4a9268]' : 'text-[#8b98a5]'}>
                                        {state === 'degraded' ? '✓' : state === 'incident_freeze' || state === 'halted' ? '✗' : '—'}
                                    </span>
                                    Auto-Recovery
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>

            {/* Timeline */}
            <Card title="State Transition Timeline">
                <div className="flex gap-2 mb-4">
                    {['all', 'normal', 'degraded', 'incident_freeze', 'halted'].map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${filter === f
                                    ? 'bg-[#5b7a9d] text-white'
                                    : 'bg-[#1a1f26] text-[#8b98a5] hover:bg-[#2d3640]'
                                }`}
                        >
                            {f === 'all' ? 'All' : f.replace('_', ' ').toUpperCase()}
                        </button>
                    ))}
                </div>

                <div className="relative">
                    {/* Timeline line */}
                    <div className="absolute left-6 top-0 bottom-0 w-px bg-[#2d3640]" />

                    <div className="space-y-4">
                        {filteredTimeline.length === 0 ? (
                            <p className="text-[#8b98a5] text-center py-8">
                                No transitions match the selected filter
                            </p>
                        ) : (
                            filteredTimeline.map((event, i) => (
                                <div key={i} className="relative flex gap-4 pl-12">
                                    {/* Timeline dot */}
                                    <div className={`absolute left-4 w-5 h-5 rounded-full border-2 ${event.to_state === 'normal' ? 'border-[#4a9268] bg-[#4a9268]/20' :
                                            event.to_state === 'degraded' ? 'border-[#c9a227] bg-[#c9a227]/20' :
                                                'border-[#e06c75] bg-[#e06c75]/20'
                                        }`} />

                                    <div className="flex-1 p-4 bg-[#0f1419] rounded-lg border border-[#2d3640] hover:border-[#3d4650] transition-colors">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2">
                                                <Badge variant={getStateBadgeVariant(event.from_state)} size="sm">
                                                    {event.from_state.replace('_', ' ').toUpperCase()}
                                                </Badge>
                                                <svg className="w-4 h-4 text-[#8b98a5]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                                </svg>
                                                <Badge variant={getStateBadgeVariant(event.to_state)} size="sm">
                                                    {event.to_state.replace('_', ' ').toUpperCase()}
                                                </Badge>
                                            </div>
                                            <span className="text-xs text-[#6b7280]">
                                                {new Date(event.timestamp).toLocaleString()}
                                            </span>
                                        </div>
                                        <p className="text-sm text-[#e7e9ea]">{event.reason}</p>
                                        <div className="flex items-center gap-4 mt-2 text-xs text-[#8b98a5]">
                                            <span>Triggered by: <span className="font-mono">{event.triggered_by}</span></span>
                                            {event.auto_recoverable && (
                                                <span className="text-[#4a9268]">Auto-recoverable</span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </Card>

            {/* Actions (only shown if not in NORMAL state) */}
            {snapshot.state !== 'normal' && (
                <Card title="Incident Actions" className="border-[#e06c75]/30">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-[#e7e9ea]">
                                Current incident requires attention
                            </p>
                            <p className="text-sm text-[#8b98a5] mt-1">
                                {snapshot.auto_recoverable
                                    ? 'This incident may auto-recover when conditions improve.'
                                    : 'This incident requires manual operator resolution.'
                                }
                            </p>
                        </div>
                        {!snapshot.auto_recoverable && (
                            <a
                                href="/operator"
                                className="px-4 py-2 bg-[#e06c75] text-white rounded-md hover:bg-[#e58089] transition-colors"
                            >
                                Go to Operator Control
                            </a>
                        )}
                    </div>
                </Card>
            )}
        </div>
    );
}
