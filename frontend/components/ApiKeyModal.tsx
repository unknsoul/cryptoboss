'use client';

/**
 * API Key Modal - Secure API key input for exchange connection
 * 
 * Features:
 * - Mode-specific fields (testnet/live)
 * - Password-type secure input
 * - Visibility toggle
 * - Validation status indicator
 * - Mode warnings for live trading
 */

import React, { useState } from 'react';
import { TradingMode, ApiConfig } from '../contexts/SessionContext';

interface ApiKeyModalProps {
    isOpen: boolean;
    targetMode: TradingMode;
    onClose: () => void;
    onSubmit: (config: ApiConfig) => Promise<boolean>;
}

export function ApiKeyModal({ isOpen, targetMode, onClose, onSubmit }: ApiKeyModalProps) {
    const [apiKey, setApiKey] = useState('');
    const [apiSecret, setApiSecret] = useState('');
    const [showKey, setShowKey] = useState(false);
    const [showSecret, setShowSecret] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [validationStatus, setValidationStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [errorMessage, setErrorMessage] = useState('');

    if (!isOpen) return null;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!apiKey || !apiSecret) {
            setErrorMessage('Both API Key and Secret are required');
            setValidationStatus('error');
            return;
        }

        setIsValidating(true);
        setValidationStatus('idle');
        setErrorMessage('');

        try {
            const success = await onSubmit({
                apiKey,
                apiSecret,
                isValidated: false,
            });

            if (success) {
                setValidationStatus('success');
                // Clear form and close after brief delay
                setTimeout(() => {
                    setApiKey('');
                    setApiSecret('');
                    onClose();
                }, 1000);
            } else {
                setValidationStatus('error');
                setErrorMessage('Failed to validate credentials');
            }
        } catch (error) {
            setValidationStatus('error');
            setErrorMessage(error instanceof Error ? error.message : 'Validation failed');
        } finally {
            setIsValidating(false);
        }
    };

    const handleClose = () => {
        setApiKey('');
        setApiSecret('');
        setValidationStatus('idle');
        setErrorMessage('');
        onClose();
    };

    const isLive = targetMode === 'live';
    const modeLabel = targetMode === 'testnet' ? 'Testnet' : 'Live';
    const endpoint = targetMode === 'testnet'
        ? 'testnet.binance.vision'
        : 'api.binance.com';

    return (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
            <div className="card max-w-md w-full">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <h2 className="heading-md">
                        {modeLabel} API Configuration
                    </h2>
                    <button
                        onClick={handleClose}
                        className="text-[#8b98a5] hover:text-[#e7e9ea]"
                    >
                        ✕
                    </button>
                </div>

                {/* Live Mode Warning */}
                {isLive && (
                    <div className="bg-[#a65454]/10 border border-[#a65454]/50 rounded-md p-4 mb-6">
                        <div className="flex items-start gap-3">
                            <span className="text-xl">⚠️</span>
                            <div>
                                <p className="text-[#a65454] font-medium">Live Trading Mode</p>
                                <p className="text-sm text-[#8b98a5] mt-1">
                                    You are connecting to the real Binance exchange.
                                    Real funds will be at risk. Proceed with caution.
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Endpoint Info */}
                <div className="bg-[#1a1f26] rounded-md p-3 mb-4">
                    <span className="label block text-xs mb-1">Exchange Endpoint</span>
                    <span className="text-[#e7e9ea] font-mono text-sm">{endpoint}</span>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* API Key */}
                    <div>
                        <label className="label block mb-2">API Key</label>
                        <div className="relative">
                            <input
                                type={showKey ? 'text' : 'password'}
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                placeholder="Enter your API key"
                                className="w-full bg-[#1a1f26] border border-[#2d3640] rounded-md px-4 py-2 text-[#e7e9ea] placeholder-[#6b7280] focus:outline-none focus:border-[#5b7a9d] pr-10"
                            />
                            <button
                                type="button"
                                onClick={() => setShowKey(!showKey)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#6b7280] hover:text-[#e7e9ea]"
                            >
                                {showKey ? '🙈' : '👁️'}
                            </button>
                        </div>
                    </div>

                    {/* API Secret */}
                    <div>
                        <label className="label block mb-2">API Secret</label>
                        <div className="relative">
                            <input
                                type={showSecret ? 'text' : 'password'}
                                value={apiSecret}
                                onChange={(e) => setApiSecret(e.target.value)}
                                placeholder="Enter your API secret"
                                className="w-full bg-[#1a1f26] border border-[#2d3640] rounded-md px-4 py-2 text-[#e7e9ea] placeholder-[#6b7280] focus:outline-none focus:border-[#5b7a9d] pr-10"
                            />
                            <button
                                type="button"
                                onClick={() => setShowSecret(!showSecret)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#6b7280] hover:text-[#e7e9ea]"
                            >
                                {showSecret ? '🙈' : '👁️'}
                            </button>
                        </div>
                    </div>

                    {/* Validation Status */}
                    {validationStatus !== 'idle' && (
                        <div className={`p-3 rounded-md ${validationStatus === 'success'
                                ? 'bg-[#4a9268]/10 border border-[#4a9268]'
                                : 'bg-[#a65454]/10 border border-[#a65454]'
                            }`}>
                            <span className={validationStatus === 'success' ? 'text-[#4a9268]' : 'text-[#a65454]'}>
                                {validationStatus === 'success'
                                    ? '✓ Credentials validated successfully'
                                    : `✗ ${errorMessage}`}
                            </span>
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-3 pt-4">
                        <button
                            type="button"
                            onClick={handleClose}
                            className="btn btn-ghost flex-1"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isValidating || !apiKey || !apiSecret}
                            className={`btn flex-1 ${isLive ? 'btn-danger' : 'btn-primary'} disabled:opacity-50`}
                        >
                            {isValidating ? 'Validating...' : `Connect to ${modeLabel}`}
                        </button>
                    </div>
                </form>

                {/* Help Text */}
                <p className="text-xs text-[#6b7280] mt-4 text-center">
                    Your keys are used only for this session and are not stored permanently.
                </p>
            </div>
        </div>
    );
}
