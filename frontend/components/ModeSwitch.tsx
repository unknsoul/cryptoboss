'use client';

import { useState } from 'react';

interface ModeSwitchProps {
    mode: 'paper' | 'live';
    onModeChange: (mode: 'paper' | 'live') => void;
}

export default function ModeSwitch({ mode, onModeChange }: ModeSwitchProps) {
    const [showConfirmation, setShowConfirmation] = useState(false);

    const handleModeSwitch = () => {
        if (mode === 'paper') {
            // Switching to live - show warning
            setShowConfirmation(true);
        } else {
            // Switching to paper - no warning needed
            onModeChange('paper');
        }
    };

    const confirmLiveMode = async () => {
        try {
            await fetch('/api/mode/live', { method: 'POST' });
            onModeChange('live');
            setShowConfirmation(false);
        } catch (error) {
            console.error('Failed to switch to live mode:', error);
            alert('Failed to switch to live mode. Check API connection.');
        }
    };

    return (
        <>
            <div className="card">
                <div className="text-text-secondary text-sm mb-2">Trading Mode</div>
                <div className="flex items-center gap-3">
                    <div
                        className={`w-3 h-3 rounded-full ${mode === 'live' ? 'bg-accent-green animate-pulse' : 'bg-accent-yellow'
                            }`}
                    />
                    <span className="font-medium text-lg">
                        {mode === 'paper' ? 'Paper Trading' : 'LIVE Trading'}
                    </span>
                    <button
                        onClick={handleModeSwitch}
                        className={`ml-auto px-3 py-1 rounded text-sm font-medium ${mode === 'paper'
                                ? 'bg-accent-green/20 text-accent-green hover:bg-accent-green/30'
                                : 'bg-accent-yellow/20 text-accent-yellow hover:bg-accent-yellow/30'
                            }`}
                    >
                        {mode === 'paper' ? 'Go Live' : 'Go Paper'}
                    </button>
                </div>
            </div>

            {/* Live Mode Confirmation Modal */}
            {showConfirmation && (
                <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
                    <div className="bg-bg-secondary border border-accent-red rounded-lg p-6 max-w-md">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="text-3xl">⚠️</div>
                            <h3 className="text-xl font-bold text-accent-red">Live Trading Warning</h3>
                        </div>

                        <div className="space-y-3 mb-6 text-text-secondary">
                            <p>You are about to enable LIVE TRADING with REAL CAPITAL.</p>

                            <div className="bg-accent-red/10 border border-accent-red rounded p-3">
                                <p className="font-medium text-accent-red text-sm">RISKS:</p>
                                <ul className="text-xs mt-2 space-y-1 list-disc list-inside">
                                    <li>You can lose ALL your capital</li>
                                    <li>No undo for live trades</li>
                                    <li>Exchange fees apply</li>
                                    <li>Market volatility can cause rapid losses</li>
                                </ul>
                            </div>

                            <p className="text-sm">
                                <strong>Have you:</strong>
                            </p>
                            <ul className="text-xs space-y-1 list-disc list-inside">
                                <li>Tested strategies in paper mode for at least 1 month?</li>
                                <li>Verified risk parameters are correct?</li>
                                <li>Set appropriate stop losses and position sizes?</li>
                                <li>Only allocated capital you can afford to lose?</li>
                            </ul>
                        </div>

                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowConfirmation(false)}
                                className="btn btn-secondary flex-1"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmLiveMode}
                                className="btn btn-danger flex-1"
                            >
                                I Understand - Go Live
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
