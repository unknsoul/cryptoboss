'use client';

/**
 * Execution & Health Page
 * 
 * CRYPTOBOSS 2.0: NO MOCK DATA
 * - All data comes from backend API
 * - Shows empty/waiting state when no data
 * - Escalation stages are reference-only (always shown)
 */

import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Escalation stage definitions (static reference, not mock data)
const ESCALATION_STAGES = [
    { level: 0, name: 'NORMAL', description: 'All systems operational', color: 'success' },
    { level: 1, name: 'DEGRADED', description: 'Increased latency or minor issues', color: 'warning' },
    { level: 2, name: 'CLOSE_ONLY', description: 'New entries blocked, exits only', color: 'warning' },
    { level: 3, name: 'HALTED', description: 'All trading suspended', color: 'danger' },
];

interface HealthData {
    exchange: {
        latency: number | null;
        status: string;
    };
    orders: {
        rejectionRate: number | null;
        partialFillRatio: number | null;
        avgFillTime: number | null;
    };
    currentStage: number;
    recentEvents: Array<{
        time: string;
        event: string;
        details: string;
        type: 'success' | 'warning' | 'danger';
    }>;
}

function HealthMetric({ label, value, unit, status }: {
    label: string;
    value: string | number | null;
    unit?: string;
    status?: 'good' | 'warning' | 'bad';
}) {
    const statusColors = {
        good: 'text-[#4a9268]',
        warning: 'text-[#c4a052]',
        bad: 'text-[#a65454]',
    };

    return (
        <div className="bg-[#1a1f26] rounded-md p-4 text-center">
            <span className="label block">{label}</span>
            <span className={`value-lg block mt-1 ${status ? statusColors[status] : 'text-[#e7e9ea]'}`}>
                {value !== null && value !== undefined ? (
                    <>{value}{unit && <span className="text-sm text-[#8b98a5] ml-1">{unit}</span>}</>
                ) : (
                    <span className="text-[#6b7280]">--</span>
                )}
            </span>
        </div>
    );
}

export default function ExecutionHealthPage() {
    const { activeAccount, token } = useAuth();
    const [healthData, setHealthData] = useState<HealthData | null>(null);
    const [loading, setLoading] = useState(false);

    // Fetch health data from backend
    useEffect(() => {
        if (!activeAccount || !token) {
            setHealthData(null);
            return;
        }

        const fetchHealth = async () => {
            setLoading(true);
            try {
                const res = await fetch(
                    `${API_URL}/api/v11/risk/state`,
                    { headers: { Authorization: `Bearer ${token}` } }
                );
                if (res.ok) {
                    const data = await res.json();
                    // Map backend response to HealthData shape
                    setHealthData({
                        exchange: {
                            latency: data.latency_ms ?? null,
                            status: data.exchange_status ?? 'UNKNOWN',
                        },
                        orders: {
                            rejectionRate: data.rejection_rate ?? null,
                            partialFillRatio: data.partial_fill_ratio ?? null,
                            avgFillTime: data.avg_fill_time_ms ?? null,
                        },
                        currentStage: data.escalation_level ?? 0,
                        recentEvents: data.recent_events ?? [],
                    });
                }
            } catch (error) {
                console.error('Failed to fetch health data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchHealth();
        const interval = setInterval(fetchHealth, 10000); // Poll every 10s
        return () => clearInterval(interval);
    }, [activeAccount?.exchange_account_id, token]);

    // Determine current stage
    const currentStageIdx = healthData?.currentStage ?? 0;
    const currentStage = ESCALATION_STAGES[currentStageIdx] ?? ESCALATION_STAGES[0];

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Execution & Health</h1>
                <p className="text-[#8b98a5] text-sm">
                    Infrastructure reliability monitoring
                </p>
            </div>

            {/* Current Escalation Stage */}
            <div className={`card border-l-4 ${currentStage.color === 'success' ? 'border-l-[#4a9268]' :
                currentStage.color === 'warning' ? 'border-l-[#c4a052]' : 'border-l-[#a65454]'
                }`}>
                <div className="flex items-center justify-between">
                    <div>
                        <span className="label">Current Stage</span>
                        <div className="flex items-center gap-3 mt-2">
                            <span className={`badge badge-${currentStage.color}`}>
                                {currentStage.name}
                            </span>
                            <span className="text-[#e7e9ea]">{currentStage.description}</span>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        {ESCALATION_STAGES.map((stage, idx) => (
                            <div
                                key={stage.level}
                                className={`w-3 h-3 rounded-full ${idx === currentStageIdx
                                    ? stage.color === 'success' ? 'bg-[#4a9268]' :
                                        stage.color === 'warning' ? 'bg-[#c4a052]' : 'bg-[#a65454]'
                                    : 'bg-[#2d3640]'
                                    }`}
                                title={stage.name}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Key Metrics - Shows "--" when no data */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <HealthMetric
                    label="Latency"
                    value={healthData?.exchange.latency ?? null}
                    unit="ms"
                    status={healthData?.exchange.latency != null
                        ? (healthData.exchange.latency < 100 ? 'good' :
                            healthData.exchange.latency < 300 ? 'warning' : 'bad')
                        : undefined}
                />
                <HealthMetric
                    label="Order Rejection Rate"
                    value={healthData?.orders.rejectionRate ?? null}
                    unit="%"
                    status={healthData?.orders.rejectionRate != null
                        ? (healthData.orders.rejectionRate < 2 ? 'good' :
                            healthData.orders.rejectionRate < 5 ? 'warning' : 'bad')
                        : undefined}
                />
                <HealthMetric
                    label="Partial Fill Ratio"
                    value={healthData?.orders.partialFillRatio ?? null}
                    unit="%"
                    status={healthData?.orders.partialFillRatio != null
                        ? (healthData.orders.partialFillRatio < 5 ? 'good' :
                            healthData.orders.partialFillRatio < 10 ? 'warning' : 'bad')
                        : undefined}
                />
                <HealthMetric
                    label="Avg Fill Time"
                    value={healthData?.orders.avgFillTime ?? null}
                    unit="ms"
                    status={healthData?.orders.avgFillTime != null
                        ? (healthData.orders.avgFillTime < 200 ? 'good' :
                            healthData.orders.avgFillTime < 500 ? 'warning' : 'bad')
                        : undefined}
                />
            </div>

            {/* Escalation Stages Reference */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Escalation Stages</span>
                </div>

                <div className="space-y-3">
                    {ESCALATION_STAGES.map((stage) => (
                        <div
                            key={stage.level}
                            className={`flex items-center justify-between p-3 rounded-md ${stage.level === currentStageIdx
                                ? 'bg-[#1a1f26] border border-[#2d3640]'
                                : ''
                                }`}
                        >
                            <div className="flex items-center gap-4">
                                <span className="text-[#6b7280] font-mono">L{stage.level}</span>
                                <span className={`badge badge-${stage.color}`}>{stage.name}</span>
                                <span className="text-[#8b98a5] text-sm">{stage.description}</span>
                            </div>
                            {stage.level === currentStageIdx && (
                                <span className="badge badge-neutral">CURRENT</span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Recent Events Log */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Recent Events</span>
                </div>

                {(!healthData || healthData.recentEvents.length === 0) ? (
                    <div className="text-center py-8 text-[#6b7280]">
                        No recent events
                    </div>
                ) : (
                    <div className="space-y-2">
                        {healthData.recentEvents.map((event, idx) => (
                            <div
                                key={idx}
                                className="flex items-center gap-4 py-2 border-b border-[#2d3640] last:border-0"
                            >
                                <span className="text-xs font-mono text-[#6b7280] w-20">{event.time}</span>
                                <div className={`status-dot ${event.type === 'success' ? 'status-dot-healthy' :
                                    event.type === 'warning' ? 'status-dot-warning' : 'status-dot-critical'
                                    }`} />
                                <span className="text-[#e7e9ea] text-sm">{event.event}</span>
                                <span className="text-[#8b98a5] text-sm">{event.details}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
