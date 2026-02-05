'use client';

/**
 * Drift Page - CRYPTOBOSS vFINAL
 * 
 * Purpose: Display drift/divergence metrics from backend
 * Rules:
 * - NO mock data - fetch from /api/drift
 * - Zero divergences for new accounts
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface DriftMetrics {
    total_divergences: number;
    max_divergence_score: number;
    last_check?: string;
}

interface DriftAlert {
    timestamp: string;
    decision_type: string;
    live_result: string;
    expected_result: string;
    divergence_score: number;
    severity: string;
}

const emptyMetrics: DriftMetrics = {
    total_divergences: 0,
    max_divergence_score: 0
};

export default function DriftPage() {
    const { activeAccount, token } = useAuth();
    const [metrics, setMetrics] = useState<DriftMetrics>(emptyMetrics);
    const [alerts, setAlerts] = useState<DriftAlert[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/drift`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (!response.ok) throw new Error('Failed to fetch drift data');
            const data = await response.json();
            setMetrics(data.data?.metrics || emptyMetrics);
            setAlerts(data.data?.alerts || []);
            setError(null);
        } catch (e: any) {
            console.error('Drift fetch error:', e);
            setError(e.message);
            setMetrics(emptyMetrics);
            setAlerts([]);
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        setMetrics(emptyMetrics);
        setAlerts([]);
        setLoading(true);
        fetchData();
    }, [activeAccount, fetchData]);

    useEffect(() => {
        const interval = setInterval(fetchData, 30000);
        return () => clearInterval(interval);
    }, [fetchData]);

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical': return 'bg-red-500/20 text-red-400';
            case 'warning': return 'bg-yellow-500/20 text-yellow-400';
            case 'info': return 'bg-blue-500/20 text-blue-400';
            default: return 'bg-gray-500/20 text-gray-400';
        }
    };

    return (
        <div className="p-6 space-y-6">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-1">Drift Monitor</h1>
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
                    {/* Metrics Summary */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6 text-center">
                            <div className="text-gray-400 text-sm">Total Divergences</div>
                            <div className={`text-4xl font-bold mt-2 ${metrics.total_divergences === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                                {metrics.total_divergences}
                            </div>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6 text-center">
                            <div className="text-gray-400 text-sm">Max Divergence Score</div>
                            <div className={`text-4xl font-bold mt-2 ${metrics.max_divergence_score < 0.3 ? 'text-green-400' : metrics.max_divergence_score < 0.7 ? 'text-yellow-400' : 'text-red-400'}`}>
                                {(metrics.max_divergence_score * 100).toFixed(0)}%
                            </div>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6 text-center">
                            <div className="text-gray-400 text-sm">System Status</div>
                            <div className={`text-2xl font-bold mt-2 ${metrics.total_divergences === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                                {metrics.total_divergences === 0 ? '✓ Aligned' : '⚠ Diverged'}
                            </div>
                        </div>
                    </div>

                    {/* Alerts */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                        <h2 className="text-white font-semibold mb-4">Drift Alerts</h2>
                        {alerts.length === 0 ? (
                            <div className="text-center py-8">
                                <div className="text-4xl mb-2">✓</div>
                                <div className="text-green-400 font-medium">No Drift Detected</div>
                                <div className="text-gray-500 text-sm">Live and expected results are aligned</div>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {alerts.map((alert, i) => (
                                    <div key={i} className="bg-gray-800/50 rounded-lg p-4">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className={`px-2 py-0.5 rounded text-xs ${getSeverityColor(alert.severity)}`}>
                                                {alert.severity.toUpperCase()}
                                            </span>
                                            <span className="text-gray-500 text-xs">
                                                {new Date(alert.timestamp).toLocaleString()}
                                            </span>
                                        </div>
                                        <div className="text-sm">
                                            <span className="text-gray-400">Type:</span>{' '}
                                            <span className="text-white">{alert.decision_type}</span>
                                        </div>
                                        <div className="text-sm">
                                            <span className="text-gray-400">Live:</span>{' '}
                                            <span className="text-red-400">{alert.live_result}</span>
                                            <span className="text-gray-500 mx-2">vs</span>
                                            <span className="text-gray-400">Expected:</span>{' '}
                                            <span className="text-green-400">{alert.expected_result}</span>
                                        </div>
                                        <div className="text-sm mt-1">
                                            <span className="text-gray-400">Score:</span>{' '}
                                            <span className="text-white">{(alert.divergence_score * 100).toFixed(0)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Info */}
                    <div className="p-4 bg-gray-800/50 rounded-lg text-sm text-gray-400">
                        <span className="text-blue-400">ℹ️</span>{' '}
                        Drift monitoring compares live trading decisions against expected behavior to detect anomalies.
                    </div>
                </>
            )}
        </div>
    );
}
