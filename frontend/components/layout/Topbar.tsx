'use client';

import { useState } from 'react';

interface TopbarProps {
    systemStatus: 'healthy' | 'warning' | 'critical' | 'unknown';
    tradingMode: 'paper' | 'live';
    lastDecision: string | null;
    onKillSwitch: () => void;
    onModeToggle: () => void;
    sidebarCollapsed: boolean;
}

export function Topbar({
    systemStatus,
    tradingMode,
    lastDecision,
    onKillSwitch,
    onModeToggle,
    sidebarCollapsed,
}: TopbarProps) {
    const [showKillConfirm, setShowKillConfirm] = useState(false);
    const [showModeConfirm, setShowModeConfirm] = useState(false);

    const statusColors = {
        healthy: 'bg-green-500',
        warning: 'bg-yellow-500',
        critical: 'bg-red-500',
        unknown: 'bg-gray-500',
    };

    const statusLabels = {
        healthy: 'System Healthy',
        warning: 'Degraded',
        critical: 'Critical',
        unknown: 'Unknown',
    };

    const handleKillClick = () => {
        setShowKillConfirm(true);
    };

    const confirmKill = () => {
        onKillSwitch();
        setShowKillConfirm(false);
    };

    const handleModeClick = () => {
        if (tradingMode === 'paper') {
            setShowModeConfirm(true);
        } else {
            onModeToggle();
        }
    };

    const confirmModeChange = () => {
        onModeToggle();
        setShowModeConfirm(false);
    };

    return (
        <>
            <header className={`fixed top-0 right-0 z-30 h-14 bg-[#0f1419] border-b border-[#2d3640] transition-all duration-300 ${sidebarCollapsed ? 'left-16' : 'left-56'}`}>
                <div className="flex h-full items-center justify-between px-6">
                    {/* Left: System Status */}
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <div className={`w-2.5 h-2.5 rounded-full ${statusColors[systemStatus]} animate-pulse`} />
                            <span className="text-sm text-[#8b98a5]">{statusLabels[systemStatus]}</span>
                        </div>
                    </div>

                    {/* Right: Controls */}
                    <div className="flex items-center gap-4">
                        {/* Last Decision */}
                        <div className="text-sm text-[#8b98a5]">
                            {lastDecision ? (
                                <>Last: <span className="text-white">{lastDecision}</span></>
                            ) : (
                                'No recent decisions'
                            )}
                        </div>

                        {/* Mode Toggle */}
                        <button
                            onClick={handleModeClick}
                            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${tradingMode === 'paper'
                                    ? 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
                                    : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                }`}
                        >
                            {tradingMode === 'paper' ? 'PAPER' : 'LIVE'}
                        </button>

                        {/* Kill Switch */}
                        <button
                            onClick={handleKillClick}
                            className="px-3 py-1.5 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700 transition-colors"
                        >
                            KILL SWITCH
                        </button>
                    </div>
                </div>
            </header>

            {/* Kill Switch Confirmation Modal */}
            {showKillConfirm && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
                    <div className="bg-[#1a1f26] rounded-lg p-6 max-w-md w-full mx-4 border border-[#2d3640]">
                        <h3 className="text-lg font-semibold text-white mb-2">⚠️ Confirm Kill Switch</h3>
                        <p className="text-[#8b98a5] mb-6">
                            This will immediately halt all trading activity and close all pending orders.
                            This action requires manual recovery.
                        </p>
                        <div className="flex gap-3 justify-end">
                            <button
                                onClick={() => setShowKillConfirm(false)}
                                className="px-4 py-2 rounded-md text-sm text-[#8b98a5] hover:bg-[#242b33]"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmKill}
                                className="px-4 py-2 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700"
                            >
                                Confirm Kill
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Mode Change Confirmation Modal */}
            {showModeConfirm && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
                    <div className="bg-[#1a1f26] rounded-lg p-6 max-w-md w-full mx-4 border border-[#2d3640]">
                        <h3 className="text-lg font-semibold text-white mb-2">⚠️ Switch to LIVE Mode</h3>
                        <p className="text-[#8b98a5] mb-6">
                            You are about to enable LIVE trading with real capital.
                            Make sure you understand the risks involved.
                        </p>
                        <div className="flex gap-3 justify-end">
                            <button
                                onClick={() => setShowModeConfirm(false)}
                                className="px-4 py-2 rounded-md text-sm text-[#8b98a5] hover:bg-[#242b33]"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmModeChange}
                                className="px-4 py-2 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700"
                            >
                                Enable LIVE
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
