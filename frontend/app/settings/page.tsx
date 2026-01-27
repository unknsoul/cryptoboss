'use client';

import { useState } from 'react';
import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

export default function SettingsPage() {
    const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper');
    const [showLiveConfirm, setShowLiveConfirm] = useState(false);

    const mockSettings = {
        api: { connected: true, latency: 45 },
        limits: {
            maxDailyLoss: 500,
            maxPositionSize: 2000,
            maxOpenPositions: 3,
        },
        killSwitch: false,
    };

    const handleModeChange = () => {
        if (tradingMode === 'paper') {
            setShowLiveConfirm(true);
        } else {
            setTradingMode('paper');
        }
    };

    const confirmLiveMode = () => {
        setTradingMode('live');
        setShowLiveConfirm(false);
    };

    return (
        <div>
            <PageHeader
                title="Settings"
                description="Safe configuration – dangerous actions require confirmation"
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Trading Mode */}
                <Card title="Trading Mode">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <span className="text-white font-medium">Current Mode</span>
                                <p className="text-sm text-[#8b98a5] mt-1">
                                    {tradingMode === 'paper'
                                        ? 'Paper trading with simulated orders'
                                        : 'Live trading with real capital'}
                                </p>
                            </div>
                            <Badge variant={tradingMode === 'paper' ? 'info' : 'danger'} size="md">
                                {tradingMode.toUpperCase()}
                            </Badge>
                        </div>
                        <button
                            onClick={handleModeChange}
                            className={`w-full py-2 rounded-md text-sm font-medium transition-colors ${tradingMode === 'paper'
                                    ? 'bg-red-600 text-white hover:bg-red-700'
                                    : 'bg-blue-600 text-white hover:bg-blue-700'
                                }`}
                        >
                            {tradingMode === 'paper' ? 'Switch to LIVE' : 'Switch to PAPER'}
                        </button>
                    </div>
                </Card>

                {/* API Status */}
                <Card title="API Status">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Connection</span>
                            <Badge variant={mockSettings.api.connected ? 'success' : 'danger'}>
                                {mockSettings.api.connected ? 'CONNECTED' : 'DISCONNECTED'}
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Latency</span>
                            <span className="text-white font-medium">{mockSettings.api.latency}ms</span>
                        </div>
                    </div>
                </Card>

                {/* Risk Limits (Read-only in Live) */}
                <Card title="Risk Limits" subtitle={tradingMode === 'live' ? 'Read-only in live mode' : 'Configurable in paper mode'}>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Max Daily Loss</span>
                            <span className="text-white font-medium">${mockSettings.limits.maxDailyLoss}</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Max Position Size</span>
                            <span className="text-white font-medium">${mockSettings.limits.maxPositionSize}</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[#8b98a5]">Max Open Positions</span>
                            <span className="text-white font-medium">{mockSettings.limits.maxOpenPositions}</span>
                        </div>
                        {tradingMode === 'live' && (
                            <div className="p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/30">
                                <p className="text-xs text-yellow-400">
                                    ⚠️ Risk limits are read-only in live mode for safety.
                                </p>
                            </div>
                        )}
                    </div>
                </Card>

                {/* Kill Switch */}
                <Card title="Emergency Controls">
                    <div className="space-y-4">
                        <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/30">
                            <div className="flex items-center justify-between">
                                <div>
                                    <span className="text-white font-medium">Kill Switch</span>
                                    <p className="text-xs text-[#8b98a5] mt-1">
                                        Immediately halt all trading activity
                                    </p>
                                </div>
                                <Badge variant={mockSettings.killSwitch ? 'danger' : 'neutral'}>
                                    {mockSettings.killSwitch ? 'ACTIVE' : 'OFF'}
                                </Badge>
                            </div>
                            <button className="w-full mt-4 py-2 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700 transition-colors">
                                ACTIVATE KILL SWITCH
                            </button>
                        </div>
                    </div>
                </Card>
            </div>

            {/* Live Mode Confirmation Modal */}
            {showLiveConfirm && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
                    <div className="bg-[#1a1f26] rounded-lg p-6 max-w-md w-full mx-4 border border-[#2d3640]">
                        <h3 className="text-lg font-semibold text-white mb-2">⚠️ Enable LIVE Trading</h3>
                        <p className="text-[#8b98a5] mb-4">
                            You are about to switch to LIVE mode. This will:
                        </p>
                        <ul className="text-sm text-[#8b98a5] mb-6 space-y-2">
                            <li>• Execute real orders on the exchange</li>
                            <li>• Use actual funds from your account</li>
                            <li>• Lock risk limits to read-only</li>
                        </ul>
                        <div className="flex gap-3 justify-end">
                            <button
                                onClick={() => setShowLiveConfirm(false)}
                                className="px-4 py-2 rounded-md text-sm text-[#8b98a5] hover:bg-[#242b33]"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmLiveMode}
                                className="px-4 py-2 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700"
                            >
                                I Understand, Enable LIVE
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
