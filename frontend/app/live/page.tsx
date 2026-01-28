'use client';

/**
 * Live Status Page
 * 
 * Purpose: Real-time awareness without noise
 * Rules:
 * - No flashing prices
 * - No trade buttons
 * - Compact, informative display
 */

import { useState, useEffect } from 'react';

function StatusRow({ label, value, subValue, variant = 'neutral' }: {
    label: string;
    value: string | number;
    subValue?: string;
    variant?: 'success' | 'warning' | 'danger' | 'neutral';
}) {
    const valueColors = {
        success: 'text-[#4a9268]',
        warning: 'text-[#c4a052]',
        danger: 'text-[#a65454]',
        neutral: 'text-[#e7e9ea]',
    };

    return (
        <div className="flex items-center justify-between py-3 border-b border-[#2d3640] last:border-0">
            <span className="text-[#8b98a5] text-sm">{label}</span>
            <div className="text-right">
                <span className={`font-medium ${valueColors[variant]}`}>{value}</span>
                {subValue && (
                    <span className="text-[#6b7280] text-sm ml-2">{subValue}</span>
                )}
            </div>
        </div>
    );
}

export default function LiveStatusPage() {
    // Client-only state to prevent hydration mismatch
    const [mounted, setMounted] = useState(false);
    const [lastUpdate, setLastUpdate] = useState('--:--:--');

    // Live data state (will be connected to API)
    const [liveData, setLiveData] = useState({
        price: { symbol: 'BTC/USDT', value: 0, change24h: 0 },
        positions: { open: 0, totalExposure: 0, unrealizedPnL: 0 },
        proposals: { active: 0, lastRejectedReason: 'None' },
        execution: { state: 'IDLE', pendingOrders: 0, lastFillTime: '--:--:--' },
    });

    // Only run on client to prevent hydration mismatch
    useEffect(() => {
        setMounted(true);

        // Initial data load
        setLiveData({
            price: { symbol: 'BTC/USDT', value: 89168.42, change24h: 2.34 },
            positions: { open: 2, totalExposure: 4500, unrealizedPnL: 156.80 },
            proposals: { active: 0, lastRejectedReason: 'Trade budget exhausted' },
            execution: { state: 'IDLE', pendingOrders: 0, lastFillTime: '14:32:15' },
        });

        const interval = setInterval(() => {
            setLastUpdate(new Date().toLocaleTimeString('en-GB', { hour12: false }));
        }, 5000);

        // Set initial time
        setLastUpdate(new Date().toLocaleTimeString('en-GB', { hour12: false }));

        return () => clearInterval(interval);
    }, []);

    const getPnLVariant = (pnl: number) => {
        if (pnl > 0) return 'success';
        if (pnl < 0) return 'danger';
        return 'neutral';
    };

    // Show loading state until mounted to prevent hydration mismatch
    if (!mounted) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Live Status</h1>
                    <p className="text-[#8b98a5] text-sm">Loading...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Live Status</h1>
                <p className="text-[#8b98a5] text-sm">
                    Real-time awareness — last update: {lastUpdate}
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Price Display - Compact, no flash */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Current Price</span>
                    </div>
                    <div className="flex items-baseline gap-3">
                        <span className="value-xl">${liveData.price.value.toLocaleString()}</span>
                        <span className={`text-sm ${liveData.price.change24h >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]'}`}>
                            {liveData.price.change24h >= 0 ? '+' : ''}{liveData.price.change24h}%
                        </span>
                    </div>
                    <div className="text-sm text-[#6b7280] mt-1">{liveData.price.symbol}</div>
                </div>

                {/* Execution State */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Execution State</span>
                    </div>
                    <div className="flex items-center gap-3 mb-4">
                        <div className={`status-dot ${liveData.execution.state === 'IDLE' ? 'status-dot-neutral' :
                                liveData.execution.state === 'EXECUTING' ? 'status-dot-warning' :
                                    'status-dot-healthy'
                            }`} />
                        <span className="value-md">{liveData.execution.state}</span>
                    </div>
                    <StatusRow
                        label="Pending Orders"
                        value={liveData.execution.pendingOrders}
                    />
                    <StatusRow
                        label="Last Fill"
                        value={liveData.execution.lastFillTime}
                    />
                </div>

                {/* Open Positions Summary */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Open Positions</span>
                    </div>
                    <StatusRow
                        label="Active Positions"
                        value={liveData.positions.open}
                    />
                    <StatusRow
                        label="Total Exposure"
                        value={`$${liveData.positions.totalExposure.toLocaleString()}`}
                    />
                    <StatusRow
                        label="Unrealized P&L"
                        value={`${liveData.positions.unrealizedPnL >= 0 ? '+' : ''}$${liveData.positions.unrealizedPnL.toFixed(2)}`}
                        variant={getPnLVariant(liveData.positions.unrealizedPnL)}
                    />
                </div>

                {/* Active Proposals */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Proposals</span>
                    </div>
                    <StatusRow
                        label="Active Proposals"
                        value={liveData.proposals.active}
                    />
                    <StatusRow
                        label="Last Rejection"
                        value={liveData.proposals.lastRejectedReason}
                        variant="neutral"
                    />
                </div>
            </div>

            {/* No Trade Buttons Notice */}
            <div className="card bg-[#1a1f26]">
                <div className="flex items-center gap-4 text-sm text-[#8b98a5]">
                    <span className="text-xl">ℹ️</span>
                    <span>
                        This is a monitoring view only. Manual trade execution is not available from the dashboard
                        to maintain system discipline and auditability.
                    </span>
                </div>
            </div>
        </div>
    );
}
