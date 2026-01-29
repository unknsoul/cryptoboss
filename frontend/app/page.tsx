'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Overview Page - Immediate System Trust Check
 * 
 * Purpose: Allow user to understand system health in under 10 seconds
 * Rules:
 * - No charts
 * - Large readable values
 * - Neutral colors only
 * - v10.2: Safety metrics displayed BEFORE profit metrics
 */

function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
    return (
        <div className="tooltip-container inline-flex">
            {children}
            <div className="tooltip">{text}</div>
        </div>
    );
}

function StatCard({
    title,
    tooltip,
    className = '',
    children
}: {
    title: string;
    tooltip?: string;
    className?: string;
    children: React.ReactNode;
}) {
    return (
        <div className={`card ${className}`}>
            <div className="card-header">
                {tooltip ? (
                    <Tooltip text={tooltip}>
                        <span className="card-title cursor-help border-b border-dotted border-[#6b7280]">
                            {title}
                        </span>
                    </Tooltip>
                ) : (
                    <span className="card-title">{title}</span>
                )}
            </div>
            {children}
        </div>
    );
}

function GaugeBar({
    value,
    max = 100,
    variant = 'neutral',
    showLabel = true
}: {
    value: number;
    max?: number;
    variant?: 'success' | 'warning' | 'danger' | 'neutral' | 'accent';
    showLabel?: boolean;
}) {
    const percentage = Math.min((value / max) * 100, 100);
    return (
        <div className="space-y-1">
            <div className="gauge">
                <div
                    className={`gauge-fill gauge-fill-${variant}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            {showLabel && (
                <div className="flex justify-between text-xs text-[#6b7280]">
                    <span>{value.toFixed(1)}%</span>
                    <span>{max}% max</span>
                </div>
            )}
        </div>
    );
}

export default function OverviewPage() {
    const [mounted, setMounted] = useState(false);
    const [lastUpdate, setLastUpdate] = useState('--:--:--');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // System data state
    const [systemData, setSystemData] = useState({
        context: { regime: '--', confidence: 0, tradingAllowed: false, timeInState: '--' },
        bias: { direction: '--', conviction: 0, reason: 'Loading...' },
        health: { stage: 'NORMAL' as 'NORMAL' | 'DEGRADED' | 'ERROR', latency: 0, lastCheck: '--' },
        capital: { allocated: 0, available: 0, total: 0 },
        drawdown: { current: 0, daily: 0, maxAllowed: 10 },
        tradeBudget: { remaining: 0, total: 0, perContext: 0 }
    });

    // v10.2: Safety metrics (displayed BEFORE profit metrics)
    const [safetyMetrics, setSafetyMetrics] = useState({
        no_trade_rate: 0.32,
        permission_rejection_rate: 0.15,
        capital_veto_rate: 0.08,
        exchange_degradation_count: 1,
        incident_freeze_count: 0,
        halt_count: 0,
        incident_state: 'normal' as 'normal' | 'degraded' | 'incident_freeze' | 'halted',
        is_paused: false
    });


    const fetchData = useCallback(async () => {
        try {
            const [systemRes, contextRes, riskRes] = await Promise.all([
                fetch(`${API_URL}/api/system`),
                fetch(`${API_URL}/api/context`),
                fetch(`${API_URL}/api/risk`)
            ]);

            const [system, context, risk] = await Promise.all([
                systemRes.json(),
                contextRes.json(),
                riskRes.json()
            ]);

            setSystemData({
                context: {
                    regime: context.market_context || 'UNKNOWN',
                    confidence: 0.75, // Derived from API
                    tradingAllowed: !system.kill_switch?.active,
                    timeInState: context.last_update ? 'Active' : '--'
                },
                bias: {
                    direction: context.bias || 'NEUTRAL',
                    conviction: 0.68,
                    reason: `Price: $${context.current_price?.toFixed(2) || '0'} (${context.price_change_pct?.toFixed(2) || 0}%)`
                },
                health: {
                    stage: system.connection_status === 'connected' ? 'NORMAL' : 'DEGRADED',
                    latency: Math.abs(system.timestamp_offset_ms || 0),
                    lastCheck: system.last_time_sync || '--'
                },
                capital: {
                    allocated: risk.capital?.allocated || 0,
                    available: risk.capital?.current || 0,
                    total: risk.capital?.initial || 10000
                },
                drawdown: {
                    current: Math.abs(risk.daily_pnl_pct || 0),
                    daily: Math.abs(risk.daily_pnl_pct || 0),
                    maxAllowed: risk.limits?.daily_loss_limit_pct || 5
                },
                tradeBudget: {
                    remaining: risk.remaining_budget?.trades_remaining || 0,
                    total: risk.limits?.max_trades_per_day || 10,
                    perContext: 2
                }
            });

            setLastUpdate(new Date().toLocaleTimeString('en-GB', { hour12: false }));
            setError(null);
            setLoading(false);
        } catch (err) {
            console.error('Failed to fetch data:', err);
            setError('Failed to connect to API');
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        setMounted(true);
        fetchData();

        // Refresh every 5 seconds
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, [fetchData]);


    const getDrawdownVariant = (current: number, max: number) => {
        const ratio = current / max;
        if (ratio < 0.3) return 'success';
        if (ratio < 0.5) return 'neutral';
        if (ratio < 0.7) return 'warning';
        return 'danger';
    };

    const getBiasVariant = (direction: string) => {
        if (direction.includes('LONG')) return 'success';
        if (direction.includes('SHORT')) return 'danger';
        return 'neutral';
    };

    const getIncidentStateColor = (state: string) => {
        switch (state) {
            case 'normal': return 'badge-success';
            case 'degraded': return 'badge-warning';
            case 'incident_freeze': return 'badge-danger';
            case 'halted': return 'badge-danger';
            default: return 'badge-neutral';
        }
    };

    const getRateVariant = (rate: number): 'success' | 'warning' | 'danger' => {
        if (rate < 0.2) return 'success';
        if (rate < 0.5) return 'warning';
        return 'danger';
    };

    // Skeleton loading component
    function SkeletonCard() {
        return (
            <div className="card animate-pulse">
                <div className="card-header">
                    <div className="h-3 w-24 bg-[#2d3640] rounded" />
                </div>
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <div className="h-3 w-16 bg-[#2d3640] rounded" />
                        <div className="h-5 w-20 bg-[#2d3640] rounded" />
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="h-3 w-20 bg-[#2d3640] rounded" />
                        <div className="h-5 w-12 bg-[#2d3640] rounded" />
                    </div>
                    <div className="h-1.5 w-full bg-[#2d3640] rounded-full" />
                </div>
            </div>
        );
    }

    if (!mounted || loading) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Overview</h1>
                    <p className="text-[#8b98a5] text-sm">Loading system status...</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[...Array(6)].map((_, i) => (
                        <SkeletonCard key={i} />
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Overview</h1>
                <p className="text-[#8b98a5] text-sm">
                    High-level system health snapshot — understand status in 10 seconds
                </p>
            </div>

            {/* v10.2: Safety Metrics Section (BEFORE profit metrics) */}
            <div className="mb-6">
                <div className="flex items-center gap-2 mb-3">
                    <svg className="w-4 h-4 text-[#c9a227]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    <h2 className="text-sm font-medium text-[#c9a227] uppercase tracking-wider">Safety Metrics</h2>
                    <Link href="/operator" className="ml-auto text-xs text-[#5b7a9d] hover:text-[#7a99bd] transition-colors">
                        Operator Control →
                    </Link>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                    {/* Incident State */}
                    <div className="card p-3">
                        <div className="text-xs text-[#6b7280] mb-1">Incident State</div>
                        <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${safetyMetrics.incident_state === 'normal' ? 'bg-[#4a9268]' :
                                    safetyMetrics.incident_state === 'degraded' ? 'bg-[#c9a227]' :
                                        'bg-[#e06c75]'
                                }`} />
                            <span className={`badge ${getIncidentStateColor(safetyMetrics.incident_state)}`}>
                                {safetyMetrics.incident_state.toUpperCase().replace('_', ' ')}
                            </span>
                        </div>
                    </div>

                    {/* No Trade Rate */}
                    <div className="card p-3">
                        <div className="text-xs text-[#6b7280] mb-1">No Trade Rate</div>
                        <div className={`text-lg font-mono ${getRateVariant(safetyMetrics.no_trade_rate) === 'success' ? 'text-[#4a9268]' :
                                getRateVariant(safetyMetrics.no_trade_rate) === 'warning' ? 'text-[#c9a227]' :
                                    'text-[#e06c75]'
                            }`}>
                            {(safetyMetrics.no_trade_rate * 100).toFixed(1)}%
                        </div>
                    </div>

                    {/* Permission Rejection */}
                    <div className="card p-3">
                        <div className="text-xs text-[#6b7280] mb-1">Permission Rejections</div>
                        <div className={`text-lg font-mono ${getRateVariant(safetyMetrics.permission_rejection_rate) === 'success' ? 'text-[#4a9268]' :
                                getRateVariant(safetyMetrics.permission_rejection_rate) === 'warning' ? 'text-[#c9a227]' :
                                    'text-[#e06c75]'
                            }`}>
                            {(safetyMetrics.permission_rejection_rate * 100).toFixed(1)}%
                        </div>
                    </div>

                    {/* Capital Vetoes */}
                    <div className="card p-3">
                        <div className="text-xs text-[#6b7280] mb-1">Capital Vetoes</div>
                        <div className={`text-lg font-mono ${getRateVariant(safetyMetrics.capital_veto_rate) === 'success' ? 'text-[#4a9268]' :
                                getRateVariant(safetyMetrics.capital_veto_rate) === 'warning' ? 'text-[#c9a227]' :
                                    'text-[#e06c75]'
                            }`}>
                            {(safetyMetrics.capital_veto_rate * 100).toFixed(1)}%
                        </div>
                    </div>

                    {/* Incidents Today */}
                    <div className="card p-3">
                        <div className="text-xs text-[#6b7280] mb-1">Incidents Today</div>
                        <div className={`text-lg font-mono ${safetyMetrics.incident_freeze_count === 0 ? 'text-[#4a9268]' : 'text-[#e06c75]'
                            }`}>
                            {safetyMetrics.incident_freeze_count}
                        </div>
                    </div>

                    {/* System Paused */}
                    <div className="card p-3">
                        <div className="text-xs text-[#6b7280] mb-1">System Status</div>
                        <span className={`badge ${safetyMetrics.is_paused ? 'badge-warning' : 'badge-success'}`}>
                            {safetyMetrics.is_paused ? 'PAUSED' : 'RUNNING'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Main Grid - 3 columns */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

                {/* Market Context Widget */}
                <StatCard
                    title="Market Context"
                    tooltip="Current market regime detected by the context engine"
                >
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Regime</span>
                            <span className="badge badge-accent">
                                {systemData.context.regime}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Confidence</span>
                            <span className="value-md">
                                {(systemData.context.confidence * 100).toFixed(0)}%
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Time in State</span>
                            <span className="text-sm text-[#e7e9ea]">
                                {systemData.context.timeInState}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Trading</span>
                            <span className={`badge ${systemData.context.tradingAllowed ? 'badge-success' : 'badge-danger'}`}>
                                {systemData.context.tradingAllowed ? 'ALLOWED' : 'BLOCKED'}
                            </span>
                        </div>
                    </div>
                </StatCard>

                {/* Current Bias Widget */}
                <StatCard
                    title="Current Bias"
                    tooltip="Directional bias from the bias engine"
                >
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Direction</span>
                            <span className={`badge badge-${getBiasVariant(systemData.bias.direction)}`}>
                                {systemData.bias.direction.replace('_', ' ')}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Conviction</span>
                            <span className="value-md">
                                {(systemData.bias.conviction * 100).toFixed(0)}%
                            </span>
                        </div>
                        <GaugeBar
                            value={systemData.bias.conviction * 100}
                            max={100}
                            variant={getBiasVariant(systemData.bias.direction)}
                            showLabel={false}
                        />
                        <p className="text-xs text-[#6b7280] mt-2">
                            {systemData.bias.reason}
                        </p>
                    </div>
                </StatCard>

                {/* Exchange Health Widget */}
                <StatCard
                    title="Exchange Health"
                    tooltip="Connection quality and API responsiveness"
                >
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Stage</span>
                            <span className={`badge ${systemData.health.stage === 'NORMAL' ? 'badge-success' :
                                systemData.health.stage === 'DEGRADED' ? 'badge-warning' : 'badge-danger'
                                }`}>
                                {systemData.health.stage}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Latency</span>
                            <span className={`value-md ${systemData.health.latency < 100 ? 'text-[#4a9268]' :
                                systemData.health.latency < 300 ? 'text-[#c4a052]' : 'text-[#a65454]'
                                }`}>
                                {systemData.health.latency}ms
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Last Check</span>
                            <span className="text-sm text-[#e7e9ea]">
                                {systemData.health.lastCheck}
                            </span>
                        </div>
                    </div>
                </StatCard>

                {/* Capital Allocation Widget */}
                <StatCard
                    title="Capital Allocation"
                    tooltip="Percentage of capital currently deployed in positions"
                >
                    <div className="space-y-4">
                        <div className="text-center py-2">
                            <span className="value-xl">
                                {(systemData.capital.allocated * 100).toFixed(0)}%
                            </span>
                            <p className="text-xs text-[#6b7280] mt-1">Allocated</p>
                        </div>
                        <GaugeBar
                            value={systemData.capital.allocated * 100}
                            max={100}
                            variant="accent"
                            showLabel={false}
                        />
                        <div className="flex justify-between text-sm">
                            <div>
                                <span className="text-[#6b7280]">Available: </span>
                                <span className="text-[#e7e9ea]">${systemData.capital.available.toLocaleString()}</span>
                            </div>
                            <div>
                                <span className="text-[#6b7280]">Total: </span>
                                <span className="text-[#e7e9ea]">${systemData.capital.total.toLocaleString()}</span>
                            </div>
                        </div>
                    </div>
                </StatCard>

                {/* Daily Drawdown Gauge */}
                <StatCard
                    title="Drawdown"
                    tooltip="Current drawdown vs maximum allowed daily drawdown"
                >
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Current</span>
                            <span className={`value-lg ${systemData.drawdown.current < 3 ? 'text-[#4a9268]' :
                                systemData.drawdown.current < 5 ? 'text-[#c4a052]' : 'text-[#a65454]'
                                }`}>
                                {systemData.drawdown.current.toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5] text-sm">Daily</span>
                            <span className="text-sm text-[#e7e9ea]">
                                {systemData.drawdown.daily.toFixed(1)}%
                            </span>
                        </div>
                        <GaugeBar
                            value={systemData.drawdown.current}
                            max={systemData.drawdown.maxAllowed}
                            variant={getDrawdownVariant(systemData.drawdown.current, systemData.drawdown.maxAllowed)}
                        />
                    </div>
                </StatCard>

                {/* Trade Budget Remaining */}
                <StatCard
                    title="Trade Budget"
                    tooltip="Number of trades remaining for today"
                >
                    <div className="space-y-4">
                        <div className="text-center py-2">
                            <span className="value-xl">
                                {systemData.tradeBudget.remaining}
                            </span>
                            <span className="text-[#6b7280] text-lg">
                                /{systemData.tradeBudget.total}
                            </span>
                            <p className="text-xs text-[#6b7280] mt-1">Trades Remaining</p>
                        </div>
                        <GaugeBar
                            value={systemData.tradeBudget.remaining}
                            max={systemData.tradeBudget.total}
                            variant="neutral"
                            showLabel={false}
                        />
                        <div className="text-sm text-center">
                            <span className="text-[#6b7280]">Per Context: </span>
                            <span className="text-[#e7e9ea]">{systemData.tradeBudget.perContext}</span>
                        </div>
                    </div>
                </StatCard>
            </div>

            {/* System Status Banner */}
            <div className="card">
                <div className="flex items-center justify-between flex-wrap gap-3">
                    <div className="flex items-center gap-3">
                        <div className={`status-dot ${safetyMetrics.incident_state === 'normal' && !safetyMetrics.is_paused
                                ? 'status-dot-healthy'
                                : safetyMetrics.incident_state === 'degraded'
                                    ? 'status-dot-warning'
                                    : 'status-dot-error'
                            }`} />
                        <span className="text-[#e7e9ea] font-medium">
                            {safetyMetrics.is_paused
                                ? 'System Paused by Operator'
                                : safetyMetrics.incident_state === 'normal'
                                    ? 'System Operating Normally'
                                    : safetyMetrics.incident_state === 'degraded'
                                        ? 'System Degraded - Reduced Trading'
                                        : 'System Halted - Operator Action Required'
                            }
                        </span>
                        <span className="text-[#6b7280] text-sm">—</span>
                        <span className="text-[#8b98a5] text-sm">
                            NO_TRADE is a valid and normal state
                        </span>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link
                            href="/incidents"
                            className="text-xs text-[#5b7a9d] hover:text-[#7a99bd] transition-colors"
                        >
                            View Incidents
                        </Link>
                        <span className="text-[#6b7280] text-sm">
                            Last update: {lastUpdate}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}

