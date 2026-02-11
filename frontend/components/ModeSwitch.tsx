'use client';

import { useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Environment Switch Component
 * 
 * CRYPTOBOSS 2.0: PAPER TRADING REMOVED
 * Only TESTNET and LIVE environments are supported.
 */

type Environment = 'testnet' | 'live';

interface EnvironmentSwitchProps {
    environment: Environment;
    onEnvironmentChange: (environment: Environment) => void;
}

export default function ModeSwitch({ environment, onEnvironmentChange }: EnvironmentSwitchProps) {
    const [showConfirmation, setShowConfirmation] = useState(false);

    const handleEnvironmentSwitch = () => {
        if (environment === 'testnet') {
            // Switching to live - show warning
            setShowConfirmation(true);
        } else {
            // Switching to testnet - no warning needed
            onEnvironmentChange('testnet');
        }
    };

    const confirmLiveMode = async () => {
        try {
            await fetch(`${API_URL}/api/mode/live`, { method: 'POST' });
            onEnvironmentChange('live');
            setShowConfirmation(false);
        } catch (error) {
            console.error('Failed to switch to live mode:', error);
            alert('Failed to switch to live mode. Check API connection.');
        }
    };

    return (
        <>
            <div className="card">
                <div className="text-text-secondary text-sm mb-2">Trading Environment</div>
                <div className="flex items-center gap-3">
                    <div
                        className={`w-3 h-3 rounded-full ${environment === 'live'
                            ? 'bg-accent-red animate-pulse'
                            : 'bg-accent-yellow'
                            }`}
                    />
                    <span className="font-medium text-lg">
                        {environment === 'testnet' ? '🟡 TESTNET' : '🔴 LIVE'}
                    </span>
                    <button
                        onClick={handleEnvironmentSwitch}
                        className={`ml-auto px-3 py-1 rounded text-sm font-medium ${environment === 'testnet'
                            ? 'bg-accent-red/20 text-accent-red hover:bg-accent-red/30'
                            : 'bg-accent-yellow/20 text-accent-yellow hover:bg-accent-yellow/30'
                            }`}
                    >
                        {environment === 'testnet' ? 'Switch to LIVE' : 'Switch to TESTNET'}
                    </button>
                </div>

                {/* Environment Info */}
                <div className="mt-3 text-xs text-text-secondary">
                    {environment === 'testnet' ? (
                        <span>Testing on Binance Testnet - No real money at risk</span>
                    ) : (
                        <span className="text-accent-red font-medium">
                            ⚠️ LIVE TRADING - Real money at risk
                        </span>
                    )}
                </div>
            </div>

            {/* Live Mode Confirmation Modal */}
            {showConfirmation && (
                <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
                    <div className="bg-bg-secondary border border-accent-red rounded-lg p-6 max-w-md">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="text-3xl">🔴</div>
                            <h3 className="text-xl font-bold text-accent-red">LIVE TRADING WARNING</h3>
                        </div>

                        <div className="space-y-3 mb-6 text-text-secondary">
                            <p>You are about to switch to <strong>LIVE TRADING</strong> with <strong>REAL MONEY</strong>.</p>

                            <div className="bg-accent-red/10 border border-accent-red rounded p-3">
                                <p className="font-medium text-accent-red text-sm">CRITICAL RISKS:</p>
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
                                <li>Tested strategies on TESTNET for at least 1 month?</li>
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
                                Cancel - Stay on TESTNET
                            </button>
                            <button
                                onClick={confirmLiveMode}
                                className="btn btn-danger flex-1"
                            >
                                I Understand - Go LIVE
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
