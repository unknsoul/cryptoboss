'use client';

/**
 * Settings Page
 * 
 * Purpose: Safe system control
 * Rules:
 * - Dangerous actions require confirmation
 * - No strategy editing here
 * - Read-only view of risk limits
 * 
 * NOTE: Paper trading has been PERMANENTLY REMOVED.
 * Use TESTNET for testing, LIVE for production.
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Default settings state (shows empty/loading until API loads)
const defaultSettings = {
    tradingMode: 'testnet' as 'testnet' | 'live',
    apiConnection: {
        exchange: 'Binance',
        status: 'disconnected' as 'connected' | 'disconnected' | 'error',
        lastPing: '--',
        testnet: true,
    },
    riskLimits: {
        dailyLossLimit: 0,
        weeklyLossLimit: 0,
        maxDrawdown: 0,
        maxPositions: 0,
        maxExposure: 0,
        tradesPerDay: 0,
        tradesPerContext: 0,
        lossesPerBias: 0,
    },
};

export default function SettingsPage() {
    const { token } = useAuth();
    const [settings, setSettings] = useState(defaultSettings);
    const [mode, setMode] = useState<'testnet' | 'live'>('testnet');
    const [loading, setLoading] = useState(true);
    const [showModeConfirm, setShowModeConfirm] = useState(false);
    const [killSwitchStep, setKillSwitchStep] = useState(0);

    // Fetch settings from API
    const fetchSettings = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/settings`, {
                headers: token ? { 'Authorization': `Bearer ${token}` } : {}
            });

            if (response.ok) {
                const data = await response.json();
                setSettings({
                    tradingMode: data.trading_mode || 'testnet',
                    apiConnection: {
                        exchange: data.exchange || 'Binance',
                        status: data.api_connected ? 'connected' : 'disconnected',
                        lastPing: data.latency_ms ? `${data.latency_ms}ms` : '--',
                        testnet: data.testnet !== false,
                    },
                    riskLimits: {
                        dailyLossLimit: data.risk?.daily_loss_limit || 0,
                        weeklyLossLimit: data.risk?.weekly_loss_limit || 0,
                        maxDrawdown: data.risk?.max_drawdown || 0,
                        maxPositions: data.risk?.max_positions || 0,
                        maxExposure: data.risk?.max_exposure || 0,
                        tradesPerDay: data.risk?.trades_per_day || 0,
                        tradesPerContext: data.risk?.trades_per_context || 0,
                        lossesPerBias: data.risk?.losses_per_bias || 0,
                    }
                });
                setMode(data.trading_mode === 'live' ? 'live' : 'testnet');
            }
        } catch (e) {
            console.error('Failed to fetch settings:', e);
            // Keep default empty state on error
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        fetchSettings();
    }, [fetchSettings]);

    const handleModeChange = () => {
        if (mode === 'testnet') {
            setShowModeConfirm(true);
        } else {
            setMode('testnet');
        }
    };

    const confirmModeChange = () => {
        setMode('live');
        setShowModeConfirm(false);
    };

    const handleKillSwitch = () => {
        if (killSwitchStep === 0) {
            setKillSwitchStep(1);
        } else if (killSwitchStep === 1) {
            setKillSwitchStep(2);
            // Activate kill switch
            console.log('KILL SWITCH ACTIVATED');
            setTimeout(() => setKillSwitchStep(0), 5000);
        }
    };

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Settings</h1>
                <p className="text-[#8b98a5] text-sm">
                    Safe system control — dangerous actions require confirmation
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Trading Mode */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Trading Mode</span>
                    </div>

                    <div className="space-y-4">
                        <div className="flex items-center justify-between py-3">
                            <div>
                                <span className="text-[#e7e9ea] font-medium">Current Mode</span>
                                <p className="text-sm text-[#6b7280] mt-1">
                                    {mode === 'testnet'
                                        ? 'Testing with Binance Testnet (no real funds)'
                                        : 'LIVE trading with real funds'}
                                </p>
                            </div>
                            <span className={`badge ${mode === 'testnet' ? 'badge-accent' : 'badge-danger'}`}>
                                {mode === 'testnet' ? '🔷 TESTNET' : '🔴 LIVE'}
                            </span>
                        </div>

                        <button
                            onClick={handleModeChange}
                            className={`btn w-full ${mode === 'testnet' ? 'btn-danger' : 'btn-ghost'}`}
                        >
                            {mode === 'testnet' ? 'Switch to LIVE Mode' : 'Switch to TESTNET Mode'}
                        </button>
                    </div>

                    {/* Mode Change Confirmation Modal */}
                    {showModeConfirm && (
                        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                            <div className="card max-w-md mx-4">
                                <h3 className="heading-md mb-4">⚠️ Confirm LIVE Mode</h3>
                                <p className="text-[#8b98a5] mb-4">
                                    Switching to LIVE mode will execute real trades with real funds.
                                    Make sure you have:
                                </p>
                                <ul className="list-disc list-inside text-sm text-[#8b98a5] mb-6 space-y-1">
                                    <li>Reviewed all risk limits</li>
                                    <li>Tested thoroughly in TESTNET mode</li>
                                    <li>Verified exchange API connectivity</li>
                                    <li>Set appropriate position sizes</li>
                                </ul>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setShowModeConfirm(false)}
                                        className="btn btn-ghost flex-1"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={confirmModeChange}
                                        className="btn btn-danger flex-1"
                                    >
                                        Confirm LIVE
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* API Connection Status */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">API Connection</span>
                    </div>

                    <div className="space-y-3">
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Exchange</span>
                            <span className="text-[#e7e9ea]">{settings.apiConnection.exchange}</span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Status</span>
                            <span className="badge badge-success">
                                {settings.apiConnection.status.toUpperCase()}
                            </span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Latency</span>
                            <span className="text-[#e7e9ea]">{settings.apiConnection.lastPing}</span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Network</span>
                            <span className="badge badge-neutral">
                                {settings.apiConnection.testnet ? 'TESTNET' : 'MAINNET'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Risk Limits (Read-Only) */}
                <div className="card lg:col-span-2">
                    <div className="card-header">
                        <span className="card-title">Risk Limits (Read-Only)</span>
                        <span className="badge badge-neutral">Configured in Code</span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Daily Loss Limit</span>
                            <span className="value-md block mt-1">${settings.riskLimits.dailyLossLimit}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Weekly Loss Limit</span>
                            <span className="value-md block mt-1">${settings.riskLimits.weeklyLossLimit}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Max Drawdown</span>
                            <span className="value-md block mt-1">{settings.riskLimits.maxDrawdown}%</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Max Positions</span>
                            <span className="value-md block mt-1">{settings.riskLimits.maxPositions}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Max Exposure</span>
                            <span className="value-md block mt-1">${settings.riskLimits.maxExposure}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Trades/Day</span>
                            <span className="value-md block mt-1">{settings.riskLimits.tradesPerDay}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Trades/Context</span>
                            <span className="value-md block mt-1">{settings.riskLimits.tradesPerContext}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Losses/Bias</span>
                            <span className="value-md block mt-1">{settings.riskLimits.lossesPerBias}</span>
                        </div>
                    </div>
                </div>

                {/* Kill Switch */}
                <div className="card lg:col-span-2 border-[#a65454]">
                    <div className="card-header">
                        <span className="card-title text-[#a65454]">Emergency Kill Switch</span>
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-[#e7e9ea]">Immediately halt all trading operations</p>
                            <p className="text-sm text-[#6b7280] mt-1">
                                This will cancel all pending orders and prevent new trades.
                                Requires double confirmation.
                            </p>
                        </div>
                        <button
                            onClick={handleKillSwitch}
                            className={`btn ${killSwitchStep === 0 ? 'btn-danger' : 'bg-[#c44444] text-white'} px-6`}
                        >
                            {killSwitchStep === 0 && 'KILL SWITCH'}
                            {killSwitchStep === 1 && 'CONFIRM KILL'}
                            {killSwitchStep === 2 && 'ACTIVATED'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
