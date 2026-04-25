'use client';

/**
 * Risk & Capital Page - CRYPTOBOSS vFINAL
 * 
 * Purpose: Show REAL risk metrics from backend
 * Rules:
 * - NO mock data - fetch from /api/risk
 * - Zero values for new accounts
 * - Re-fetch on account change
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface RiskData {
    dailyLoss: { current: number; limit: number; pct: number };
    weeklyLoss: { current: number; limit: number; pct: number };
    tradeBudget: {
        daily: { used: number; max: number };
        perContext: { used: number; max: number };
    };
    killSwitch: boolean;
    consecutiveLosses: number;
    currentContext: string;
}

const emptyRisk: RiskData = {
    dailyLoss: { current: 0, limit: 500, pct: 0 },
    weeklyLoss: { current: 0, limit: 1500, pct: 0 },
    tradeBudget: {
        daily: { used: 0, max: 10 },
        perContext: { used: 0, max: 3 }
    },
    killSwitch: false,
    consecutiveLosses: 0,
    currentContext: 'UNKNOWN'
};

export default function RiskCapitalPage() {
    const { activeAccount, token } = useAuth();
    const [risk, setRisk] = useState<RiskData>(emptyRisk);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchRisk = useCallback(async () => {
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/risk`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch risk data');
            }

            const payload = await response.json();
            const data: any = unwrapApiData(payload);
            const dailyLimit =
                data.limits?.daily_loss_limit ??
                ((data.limits?.daily_loss_limit_pct ?? 5) / 100) * (data.capital?.initial || 10000);
            const weeklyLimit = data.limits?.weekly_loss_limit ?? dailyLimit * 3;
            const maxDailyTrades = data.limits?.max_trades_per_day || 10;
            const tradesRemaining = data.remaining_budget?.trades_remaining;
            const tradesUsed = typeof tradesRemaining === 'number'
                ? Math.max(maxDailyTrades - tradesRemaining, 0)
                : (data.remaining_budget?.trades_today || 0);

            // Map backend response to our structure
            setRisk({
                dailyLoss: {
                    current: data.daily_pnl || 0,
                    limit: dailyLimit,
                    pct: Math.abs((data.daily_pnl || 0) / dailyLimit * 100)
                },
                weeklyLoss: {
                    current: data.weekly_pnl || 0,
                    limit: weeklyLimit,
                    pct: Math.abs((data.weekly_pnl || 0) / weeklyLimit * 100)
                },
                tradeBudget: {
                    daily: {
                        used: tradesUsed,
                        max: maxDailyTrades
                    },
                    perContext: {
                        used: data.remaining_budget?.context_trades || 0,
                        max: 3
                    }
                },
                killSwitch: data.kill_switch_active || false,
                consecutiveLosses: data.consecutive_losses || 0,
                currentContext: data.current_context || 'UNKNOWN'
            });
            setError(null);
        } catch (e: any) {
            console.error('Risk fetch error:', e);
            setError(e.message);
            setRisk(emptyRisk);
        } finally {
            setLoading(false);
        }
    }, [token]);

    // Fetch on mount and when account changes
    useEffect(() => {
        setRisk(emptyRisk);
        setLoading(true);
        fetchRisk();
    }, [activeAccount, fetchRisk]);

    // Refresh every 10 seconds
    useEffect(() => {
        const interval = setInterval(fetchRisk, 10000);
        return () => clearInterval(interval);
    }, [fetchRisk]);

    const getLossColor = (pct: number) => {
        if (pct < 50) return 'bg-green-500';
        if (pct < 75) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-1">Risk & Capital</h1>
                <p className="text-gray-400 text-sm">
                    {activeAccount
                        ? `Account: ${activeAccount.label}`
                        : 'No account selected'}
                </p>
            </div>

            {/* Loading State */}
            {loading && (
                <div className="text-center py-12 text-gray-400">Loading risk data...</div>
            )}

            {/* Error State */}
            {error && (
                <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-400">
                    Error: {error}
                </div>
            )}

            {/* No Account */}
            {!loading && !activeAccount && (
                <div className="text-center py-12">
                    <div className="text-5xl mb-4">🔐</div>
                    <div className="text-xl text-white mb-2">No Account Selected</div>
                    <div className="text-gray-400">
                        Please select an exchange account to view risk metrics
                    </div>
                </div>
            )}

            {/* Has Account - Show Data */}
            {!loading && activeAccount && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                    {/* Daily Loss Limit */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                        <h2 className="text-white font-semibold mb-4">Daily Loss Limit</h2>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-gray-400">Current P&L</span>
                                <span className={`text-xl font-bold ${risk.dailyLoss.current >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    ${risk.dailyLoss.current.toFixed(2)}
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-gray-400">Loss Limit</span>
                                <span className="text-white font-medium">-${risk.dailyLoss.limit}</span>
                            </div>
                            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                    className={`h-full transition-all ${getLossColor(risk.dailyLoss.pct)}`}
                                    style={{ width: `${Math.min(risk.dailyLoss.pct, 100)}%` }}
                                />
                            </div>
                            <div className="text-center text-sm text-gray-500">
                                {risk.dailyLoss.pct.toFixed(0)}% of limit used
                            </div>
                        </div>
                    </div>

                    {/* Trade Budget */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                        <h2 className="text-white font-semibold mb-4">Trade Budget</h2>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-gray-400">Daily Trades</span>
                                <span className={`px-3 py-1 rounded text-sm font-medium ${risk.tradeBudget.daily.used < risk.tradeBudget.daily.max
                                        ? 'bg-green-500/20 text-green-400'
                                        : 'bg-red-500/20 text-red-400'
                                    }`}>
                                    {risk.tradeBudget.daily.used} / {risk.tradeBudget.daily.max}
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-gray-400">Per Context</span>
                                <span className={`px-3 py-1 rounded text-sm font-medium ${risk.tradeBudget.perContext.used < risk.tradeBudget.perContext.max
                                        ? 'bg-green-500/20 text-green-400'
                                        : 'bg-yellow-500/20 text-yellow-400'
                                    }`}>
                                    {risk.tradeBudget.perContext.used} / {risk.tradeBudget.perContext.max}
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-gray-400">Consecutive Losses</span>
                                <span className={`px-3 py-1 rounded text-sm font-medium ${risk.consecutiveLosses < 2
                                        ? 'bg-green-500/20 text-green-400'
                                        : 'bg-yellow-500/20 text-yellow-400'
                                    }`}>
                                    {risk.consecutiveLosses}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Kill Switch Status */}
                    <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6 lg:col-span-2">
                        <h2 className="text-white font-semibold mb-4">Kill Switch Status</h2>
                        <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                            <div className="flex items-center gap-4">
                                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${risk.killSwitch ? 'bg-red-500/20' : 'bg-green-500/20'
                                    }`}>
                                    {risk.killSwitch ? (
                                        <span className="text-red-400 text-2xl">✗</span>
                                    ) : (
                                        <span className="text-green-400 text-2xl">✓</span>
                                    )}
                                </div>
                                <div>
                                    <span className="text-white font-medium block">
                                        {risk.killSwitch ? 'KILL SWITCH ACTIVE' : 'System Normal'}
                                    </span>
                                    <p className="text-sm text-gray-500">
                                        {risk.killSwitch
                                            ? 'All trading halted. Manual recovery required.'
                                            : 'Trading operations enabled.'}
                                    </p>
                                </div>
                            </div>
                            <span className={`px-4 py-2 rounded-lg font-bold ${risk.killSwitch
                                    ? 'bg-red-500/20 text-red-400'
                                    : 'bg-green-500/20 text-green-400'
                                }`}>
                                {risk.killSwitch ? 'HALTED' : 'ACTIVE'}
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
