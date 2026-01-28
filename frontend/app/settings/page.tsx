'use client';

/**
 * Settings Page
 * 
 * Purpose: Safe system control
 * Rules:
 * - Dangerous actions require confirmation
 * - No strategy editing here
 * - Read-only view of risk limits
 */

import { useState } from 'react';

// Mock settings data
const settingsData = {
    tradingMode: 'paper' as 'paper' | 'live',
    apiConnection: {
        exchange: 'Binance',
        status: 'connected',
        lastPing: '45ms',
        testnet: true,
    },
    riskLimits: {
        dailyLossLimit: 500,
        weeklyLossLimit: 1500,
        maxDrawdown: 10,
        maxPositions: 5,
        maxExposure: 5000,
        tradesPerDay: 10,
        tradesPerContext: 3,
        lossesPerBias: 2,
    },
};

export default function SettingsPage() {
    const [mode, setMode] = useState(settingsData.tradingMode);
    const [showModeConfirm, setShowModeConfirm] = useState(false);
    const [killSwitchStep, setKillSwitchStep] = useState(0);

    const handleModeChange = () => {
        if (mode === 'paper') {
            setShowModeConfirm(true);
        } else {
            setMode('paper');
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
                                    {mode === 'paper'
                                        ? 'Simulated trading with no real funds at risk'
                                        : 'LIVE trading with real funds'}
                                </p>
                            </div>
                            <span className={`badge ${mode === 'paper' ? 'badge-accent' : 'badge-danger'}`}>
                                {mode === 'paper' ? '📄 PAPER' : '🔴 LIVE'}
                            </span>
                        </div>

                        <button
                            onClick={handleModeChange}
                            className={`btn w-full ${mode === 'paper' ? 'btn-danger' : 'btn-ghost'}`}
                        >
                            {mode === 'paper' ? 'Switch to LIVE Mode' : 'Switch to PAPER Mode'}
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
                                    <li>Tested thoroughly in paper mode</li>
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
                            <span className="text-[#e7e9ea]">{settingsData.apiConnection.exchange}</span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Status</span>
                            <span className="badge badge-success">
                                {settingsData.apiConnection.status.toUpperCase()}
                            </span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Latency</span>
                            <span className="text-[#e7e9ea]">{settingsData.apiConnection.lastPing}</span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                            <span className="text-[#8b98a5]">Network</span>
                            <span className="badge badge-neutral">
                                {settingsData.apiConnection.testnet ? 'TESTNET' : 'MAINNET'}
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
                            <span className="value-md block mt-1">${settingsData.riskLimits.dailyLossLimit}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Weekly Loss Limit</span>
                            <span className="value-md block mt-1">${settingsData.riskLimits.weeklyLossLimit}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Max Drawdown</span>
                            <span className="value-md block mt-1">{settingsData.riskLimits.maxDrawdown}%</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Max Positions</span>
                            <span className="value-md block mt-1">{settingsData.riskLimits.maxPositions}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Max Exposure</span>
                            <span className="value-md block mt-1">${settingsData.riskLimits.maxExposure}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Trades/Day</span>
                            <span className="value-md block mt-1">{settingsData.riskLimits.tradesPerDay}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Trades/Context</span>
                            <span className="value-md block mt-1">{settingsData.riskLimits.tradesPerContext}</span>
                        </div>
                        <div className="bg-[#1a1f26] rounded-md p-4">
                            <span className="label block">Losses/Bias</span>
                            <span className="value-md block mt-1">{settingsData.riskLimits.lossesPerBias}</span>
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
