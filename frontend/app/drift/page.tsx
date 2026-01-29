'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/shared/Card';
import { Badge } from '@/components/shared/Badge';
import { PageHeader } from '@/components/shared/PageHeader';
import { StatusDot } from '@/components/shared/StatusDot';

// Mock data
const mockMetrics = {
    total_comparisons: 15420,
    total_divergences: 3,
    drift_rate: 0.0002,
    last_divergence: new Date(Date.now() - 7200000).toISOString(),
    divergences_by_type: {
        'context': 1,
        'bias': 1,
        'permission': 1
    },
    affected_modules_count: {
        'market_context': 1,
        'bias_engine': 1,
        'trade_permission': 1
    }
};

const mockAlerts = [
    {
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        decision_type: 'permission',
        live_result: 'REJECTED',
        expected_result: 'APPROVED',
        divergence_score: 1.0,
        affected_modules: ['trade_permission', 'risk_guardian'],
        severity: 'warning',
        context: { symbol: 'BTC/USDT', reason: 'Timing race condition' }
    },
    {
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        decision_type: 'bias',
        live_result: 'LONG_BIAS',
        expected_result: 'NEUTRAL',
        divergence_score: 0.5,
        affected_modules: ['bias_engine'],
        severity: 'info',
        context: { symbol: 'ETH/USDT', reason: 'Indicator timing' }
    },
    {
        timestamp: new Date(Date.now() - 172800000).toISOString(),
        decision_type: 'context',
        live_result: 'TRENDING',
        expected_result: 'RANGING',
        divergence_score: 0.7,
        affected_modules: ['market_context'],
        severity: 'warning',
        context: { symbol: 'BTC/USDT', reason: 'Volatility calculation difference' }
    }
];

type DriftMetrics = typeof mockMetrics;
type DriftAlert = typeof mockAlerts[0];

export default function DriftPage() {
    const [metrics, setMetrics] = useState<DriftMetrics>(mockMetrics);
    const [alerts, setAlerts] = useState<DriftAlert[]>(mockAlerts);
    const [threshold, setThreshold] = useState(0.01);

    const isDrifting = metrics.drift_rate > threshold;
    const driftPercentage = (metrics.drift_rate * 100).toFixed(4);
    const thresholdPercentage = (threshold * 100).toFixed(2);

    const getSeverityColor = (severity: string): 'success' | 'warning' | 'danger' | 'info' => {
        switch (severity) {
            case 'critical': return 'danger';
            case 'warning': return 'warning';
            case 'info': return 'info';
            default: return 'success';
        }
    };

    const getTypeColor = (type: string): string => {
        switch (type) {
            case 'context': return 'text-[#61afef]';
            case 'bias': return 'text-[#c678dd]';
            case 'permission': return 'text-[#e5c07b]';
            case 'trade': return 'text-[#e06c75]';
            default: return 'text-[#8b98a5]';
        }
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Decision Drift Monitor"
                description="Live vs replay decision comparison - detect behavioral divergence"
            />

            {/* Drift Status */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Drift Rate Gauge */}
                <Card className="md:col-span-2">
                    <div className="flex items-center gap-6">
                        {/* Gauge */}
                        <div className="relative w-32 h-32">
                            <svg className="w-full h-full transform -rotate-90">
                                {/* Background circle */}
                                <circle
                                    cx="64"
                                    cy="64"
                                    r="56"
                                    stroke="#2d3640"
                                    strokeWidth="8"
                                    fill="none"
                                />
                                {/* Progress circle */}
                                <circle
                                    cx="64"
                                    cy="64"
                                    r="56"
                                    stroke={isDrifting ? '#e06c75' : '#4a9268'}
                                    strokeWidth="8"
                                    fill="none"
                                    strokeLinecap="round"
                                    strokeDasharray={`${Math.min(metrics.drift_rate / threshold, 1) * 351.86} 351.86`}
                                    className="transition-all duration-500"
                                />
                                {/* Threshold marker */}
                                <circle
                                    cx="64"
                                    cy="8"
                                    r="3"
                                    fill="#c9a227"
                                    className="transform rotate-90 origin-[64px_64px]"
                                    style={{ transform: `rotate(${360}deg)` }}
                                />
                            </svg>
                            <div className="absolute inset-0 flex items-center justify-center flex-col">
                                <span className={`text-2xl font-bold ${isDrifting ? 'text-[#e06c75]' : 'text-[#4a9268]'}`}>
                                    {driftPercentage}%
                                </span>
                                <span className="text-xs text-[#8b98a5]">drift rate</span>
                            </div>
                        </div>

                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                                <StatusDot status={isDrifting ? 'danger' : 'success'} pulse={isDrifting} />
                                <span className="text-lg font-medium text-[#e7e9ea]">
                                    {isDrifting ? 'DRIFT DETECTED' : 'WITHIN THRESHOLD'}
                                </span>
                            </div>
                            <p className="text-sm text-[#8b98a5] mb-3">
                                Threshold: {thresholdPercentage}% | Target: &lt;1%
                            </p>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-[#8b98a5]">Comparisons:</span>
                                <span className="text-sm font-mono text-[#e7e9ea]">
                                    {metrics.total_comparisons.toLocaleString()}
                                </span>
                            </div>
                        </div>
                    </div>
                </Card>

                <Card title="Total Divergences">
                    <div className="text-3xl font-bold text-[#e7e9ea]">
                        {metrics.total_divergences}
                    </div>
                    <p className="text-sm text-[#8b98a5] mt-1">
                        Last: {metrics.last_divergence
                            ? new Date(metrics.last_divergence).toLocaleString()
                            : 'Never'
                        }
                    </p>
                </Card>

                <Card title="Divergence by Type">
                    <div className="space-y-2">
                        {Object.entries(metrics.divergences_by_type).map(([type, count]) => (
                            <div key={type} className="flex items-center justify-between">
                                <span className={`text-sm ${getTypeColor(type)}`}>
                                    {type}
                                </span>
                                <span className="text-sm font-mono text-[#e7e9ea]">{count}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>

            {/* Affected Modules */}
            <Card title="Affected Modules">
                <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                    {Object.entries(metrics.affected_modules_count).map(([module, count]) => (
                        <div
                            key={module}
                            className="p-3 bg-[#0f1419] rounded-lg border border-[#2d3640]"
                        >
                            <div className="text-lg font-bold text-[#e7e9ea]">{count}</div>
                            <div className="text-xs text-[#8b98a5] truncate">{module}</div>
                        </div>
                    ))}
                </div>
            </Card>

            {/* Recent Alerts */}
            <Card title="Recent Drift Alerts" subtitle="Last 50 alerts">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-[#2d3640]">
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Time</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Type</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Live Result</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Expected</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Score</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Severity</th>
                                <th className="text-left py-3 px-4 text-sm font-medium text-[#8b98a5]">Modules</th>
                            </tr>
                        </thead>
                        <tbody>
                            {alerts.length === 0 ? (
                                <tr>
                                    <td colSpan={7} className="py-8 text-center text-[#8b98a5]">
                                        No drift alerts recorded
                                    </td>
                                </tr>
                            ) : (
                                alerts.map((alert, i) => (
                                    <tr key={i} className="border-b border-[#2d3640]/50 hover:bg-[#1a1f26] transition-colors">
                                        <td className="py-3 px-4 text-sm text-[#8b98a5]">
                                            {new Date(alert.timestamp).toLocaleString()}
                                        </td>
                                        <td className="py-3 px-4">
                                            <span className={`text-sm font-medium ${getTypeColor(alert.decision_type)}`}>
                                                {alert.decision_type}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4">
                                            <code className="text-sm text-[#e06c75] bg-[#e06c75]/10 px-2 py-0.5 rounded">
                                                {alert.live_result}
                                            </code>
                                        </td>
                                        <td className="py-3 px-4">
                                            <code className="text-sm text-[#4a9268] bg-[#4a9268]/10 px-2 py-0.5 rounded">
                                                {alert.expected_result}
                                            </code>
                                        </td>
                                        <td className="py-3 px-4">
                                            <div className="flex items-center gap-2">
                                                <div className="w-16 h-1.5 bg-[#2d3640] rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full ${alert.divergence_score >= 0.8 ? 'bg-[#e06c75]' :
                                                                alert.divergence_score >= 0.5 ? 'bg-[#c9a227]' :
                                                                    'bg-[#4a9268]'
                                                            }`}
                                                        style={{ width: `${alert.divergence_score * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-xs text-[#8b98a5]">
                                                    {(alert.divergence_score * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="py-3 px-4">
                                            <Badge variant={getSeverityColor(alert.severity)} size="sm">
                                                {alert.severity.toUpperCase()}
                                            </Badge>
                                        </td>
                                        <td className="py-3 px-4">
                                            <div className="flex flex-wrap gap-1">
                                                {alert.affected_modules.map((mod, j) => (
                                                    <span
                                                        key={j}
                                                        className="text-xs text-[#8b98a5] bg-[#2d3640] px-1.5 py-0.5 rounded"
                                                    >
                                                        {mod}
                                                    </span>
                                                ))}
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </Card>

            {/* Threshold Configuration */}
            <Card title="Threshold Configuration" subtitle="Read-only in LIVE mode">
                <div className="flex items-center gap-4">
                    <div className="flex-1">
                        <label className="block text-sm font-medium text-[#8b98a5] mb-2">
                            Drift Rate Threshold
                        </label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="0.001"
                                max="0.05"
                                step="0.001"
                                value={threshold}
                                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                className="flex-1 h-2 bg-[#2d3640] rounded-lg appearance-none cursor-pointer"
                                disabled
                            />
                            <span className="text-lg font-mono text-[#e7e9ea] min-w-[60px]">
                                {thresholdPercentage}%
                            </span>
                        </div>
                        <p className="text-xs text-[#6b7280] mt-2">
                            System alerts when drift rate exceeds this threshold. Recommended: 1%
                        </p>
                    </div>
                    <div className="text-right">
                        <Badge variant="warning">LIVE MODE</Badge>
                        <p className="text-xs text-[#8b98a5] mt-1">Config read-only</p>
                    </div>
                </div>
            </Card>
        </div>
    );
}
