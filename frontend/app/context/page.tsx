'use client';

/**
 * Market Context Page
 * 
 * CRYPTOBOSS 2.0: NO MOCK DATA
 * - All data comes from backend API
 * - Shows empty/waiting state when no data
 * - Timeline view of market regimes
 */

import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const stateColors: Record<string, string> = {
    'TRENDING_UP': 'badge-success',
    'TRENDING_DOWN': 'badge-danger',
    'RANGING': 'badge-accent',
    'HIGH_VOLATILITY': 'badge-warning',
    'LOW_VOLATILITY': 'badge-neutral',
};

const stateIcons: Record<string, string> = {
    'TRENDING_UP': '📈',
    'TRENDING_DOWN': '📉',
    'RANGING': '↔️',
    'HIGH_VOLATILITY': '⚡',
    'LOW_VOLATILITY': '😴',
};

interface ContextData {
    current: {
        state: string;
        confidence: number;
        timeInState: string;
        tradingAllowed: boolean;
    };
    cooldown: {
        active: boolean;
        remaining: string;
        reason: string | null;
    };
    history: Array<{
        state: string;
        startTime: string;
        duration: string;
        active: boolean;
        transitionReason: string;
    }>;
}

export default function MarketContextPage() {
    const { activeAccount, token } = useAuth();
    const [contextData, setContextData] = useState<ContextData | null>(null);
    const [loading, setLoading] = useState(false);

    // Fetch context data from backend
    useEffect(() => {
        if (!activeAccount || !token) {
            setContextData(null);
            return;
        }

        const fetchContext = async () => {
            setLoading(true);
            try {
                const res = await fetch(
                    `${API_URL}/api/v11/risk/state`,
                    { headers: { Authorization: `Bearer ${token}` } }
                );
                if (res.ok) {
                    const data = await res.json();
                    // Only set if data has context info
                    if (data.market_context || data.regime) {
                        setContextData({
                            current: {
                                state: data.market_context?.state ?? data.regime ?? 'UNKNOWN',
                                confidence: data.market_context?.confidence ?? 0,
                                timeInState: data.market_context?.time_in_state ?? '--',
                                tradingAllowed: data.market_context?.trading_allowed ?? false,
                            },
                            cooldown: {
                                active: data.cooldown?.active ?? false,
                                remaining: data.cooldown?.remaining ?? '0m',
                                reason: data.cooldown?.reason ?? null,
                            },
                            history: data.market_context?.history ?? [],
                        });
                    }
                }
            } catch (error) {
                console.error('Failed to fetch context:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchContext();
        const interval = setInterval(fetchContext, 5000);
        return () => clearInterval(interval);
    }, [activeAccount?.exchange_account_id, token]);

    // Show empty state when no data
    if (!contextData) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Market Context</h1>
                    <p className="text-[#8b98a5] text-sm">
                        Market regime understanding — timeline view
                    </p>
                </div>

                {/* Empty current state */}
                <div className="card border-l-4 border-l-[#6b7280]">
                    <div>
                        <span className="label">Current Regime</span>
                        <div className="flex items-center gap-3 mt-2">
                            <span className="badge badge-neutral">WAITING</span>
                            <span className="text-[#6b7280]">
                                {activeAccount
                                    ? 'Waiting for market data...'
                                    : 'Select an exchange account to view market context'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Empty metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                        <span className="label block">Confidence</span>
                        <span className="value-lg block mt-1 text-[#6b7280]">--</span>
                    </div>
                    <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                        <span className="label block">Time in State</span>
                        <span className="value-lg block mt-1 text-[#6b7280]">--</span>
                    </div>
                    <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                        <span className="label block">Trading</span>
                        <span className="value-lg block mt-1 text-[#6b7280]">--</span>
                    </div>
                    <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                        <span className="label block">Cooldown</span>
                        <span className="value-lg block mt-1 text-[#6b7280]">--</span>
                    </div>
                </div>

                {/* Empty timeline */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Regime Timeline</span>
                    </div>
                    <div className="text-center py-8 text-[#6b7280]">
                        No regime history available
                    </div>
                </div>
            </div>
        );
    }

    // Main view with data
    const currentIcon = stateIcons[contextData.current.state] || '❓';
    const currentColor = stateColors[contextData.current.state] || 'badge-neutral';

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Market Context</h1>
                <p className="text-[#8b98a5] text-sm">
                    Market regime understanding — timeline view
                </p>
            </div>

            {/* Current Regime */}
            <div className={`card border-l-4 ${contextData.current.tradingAllowed ? 'border-l-[#4a9268]' : 'border-l-[#c4a052]'
                }`}>
                <div className="flex items-center justify-between">
                    <div>
                        <span className="label">Current Regime</span>
                        <div className="flex items-center gap-3 mt-2">
                            <span className="text-2xl">{currentIcon}</span>
                            <span className={`badge ${currentColor}`}>
                                {contextData.current.state}
                            </span>
                            <span className="text-[#e7e9ea]">
                                for {contextData.current.timeInState}
                            </span>
                        </div>
                    </div>
                    <div className="text-right">
                        <span className={`badge ${contextData.current.tradingAllowed ? 'badge-success' : 'badge-warning'}`}>
                            {contextData.current.tradingAllowed ? 'Trading Allowed' : 'Trading Paused'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                    <span className="label block">Confidence</span>
                    <span className="value-lg block mt-1 text-[#e7e9ea]">
                        {contextData.current.confidence}%
                    </span>
                </div>
                <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                    <span className="label block">Time in State</span>
                    <span className="value-lg block mt-1 text-[#e7e9ea]">
                        {contextData.current.timeInState}
                    </span>
                </div>
                <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                    <span className="label block">Trading</span>
                    <span className={`value-lg block mt-1 ${contextData.current.tradingAllowed ? 'text-[#4a9268]' : 'text-[#c4a052]'}`}>
                        {contextData.current.tradingAllowed ? 'OPEN' : 'PAUSED'}
                    </span>
                </div>
                <div className="bg-[#1a1f26] rounded-md p-4 text-center">
                    <span className="label block">Cooldown</span>
                    <span className={`value-lg block mt-1 ${contextData.cooldown.active ? 'text-[#c4a052]' : 'text-[#4a9268]'}`}>
                        {contextData.cooldown.active ? contextData.cooldown.remaining : 'None'}
                    </span>
                </div>
            </div>

            {/* Cooldown Alert */}
            {contextData.cooldown.active && (
                <div className="card bg-[#c4a052]/10 border border-[#c4a052]/50">
                    <div className="flex items-center gap-3">
                        <span className="text-xl">⏸️</span>
                        <div>
                            <span className="text-[#c4a052] font-medium">Cooldown Active</span>
                            <p className="text-sm text-[#8b98a5]">
                                {contextData.cooldown.reason || 'Regime transition cooldown'} — {contextData.cooldown.remaining} remaining
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Regime Timeline */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Regime Timeline</span>
                </div>

                {contextData.history.length === 0 ? (
                    <div className="text-center py-8 text-[#6b7280]">
                        No regime history available
                    </div>
                ) : (
                    <div className="space-y-0">
                        {contextData.history.map((entry, idx) => (
                            <div
                                key={idx}
                                className={`flex items-start gap-4 p-4 ${entry.active ? 'bg-[#1a1f26] rounded-md' : ''}`}
                            >
                                {/* Timeline connector */}
                                <div className="flex flex-col items-center">
                                    <div className={`w-2.5 h-2.5 rounded-full ${entry.active ? 'bg-[#5b7a9d]' : 'bg-[#2d3640]'}`} />
                                    {idx < contextData.history.length - 1 && (
                                        <div className="w-px h-12 bg-[#2d3640]" />
                                    )}
                                </div>

                                {/* Content */}
                                <div className="flex-1">
                                    <div className="flex items-center gap-3 mb-1">
                                        <span className="text-xs text-[#6b7280]">{entry.startTime}</span>
                                        <span className={`badge ${stateColors[entry.state] || 'badge-neutral'}`}>
                                            {stateIcons[entry.state] || ''} {entry.state}
                                        </span>
                                        <span className="text-xs text-[#6b7280]">{entry.duration}</span>
                                        {entry.active && <span className="badge badge-neutral">NOW</span>}
                                    </div>
                                    <p className="text-sm text-[#8b98a5]">{entry.transitionReason}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
