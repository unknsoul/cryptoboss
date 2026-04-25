'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiKeyModal } from '../../components/ApiKeyModal';
import { TradingMode, useSession } from '../../contexts/SessionContext';
import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';
type EngineStatus = 'running' | 'paused' | 'stopped';
type NoticeTone = 'success' | 'error';

interface RiskLimits {
    dailyLossLimit: number;
    weeklyLossLimit: number;
    maxDrawdown: number;
    maxPositions: number;
    maxExposure: number;
    tradesPerDay: number;
    tradesPerContext: number;
    lossesPerBias: number;
}

interface SettingsState {
    tradingMode: TradingMode;
    engineStatus: EngineStatus;
    apiConnection: {
        exchange: string;
        status: ConnectionStatus;
        lastPing: string;
        testnet: boolean;
    };
    riskLimits: RiskLimits;
}

const defaultRiskLimits: RiskLimits = {
    dailyLossLimit: 0,
    weeklyLossLimit: 0,
    maxDrawdown: 0,
    maxPositions: 0,
    maxExposure: 0,
    tradesPerDay: 0,
    tradesPerContext: 0,
    lossesPerBias: 0,
};

const defaultSettings: SettingsState = {
    tradingMode: 'testnet',
    engineStatus: 'stopped',
    apiConnection: {
        exchange: 'Binance',
        status: 'disconnected',
        lastPing: '--',
        testnet: true,
    },
    riskLimits: defaultRiskLimits,
};

const riskFieldMeta: Array<{
    key: keyof RiskLimits;
    label: string;
    suffix?: string;
    step?: number;
    min?: number;
}> = [
        { key: 'dailyLossLimit', label: 'Daily Loss Limit', step: 10, min: 1 },
        { key: 'weeklyLossLimit', label: 'Weekly Loss Limit', step: 10, min: 1 },
        { key: 'maxDrawdown', label: 'Max Drawdown', suffix: '%', step: 0.1, min: 0.1 },
        { key: 'maxPositions', label: 'Max Positions', step: 1, min: 1 },
        { key: 'maxExposure', label: 'Max Exposure', step: 10, min: 1 },
        { key: 'tradesPerDay', label: 'Trades/Day', step: 1, min: 1 },
        { key: 'tradesPerContext', label: 'Trades/Context', step: 1, min: 1 },
        { key: 'lossesPerBias', label: 'Losses/Bias', step: 1, min: 1 },
    ];

function mapRiskLimits(risk: any): RiskLimits {
    return {
        dailyLossLimit: Number(risk?.daily_loss_limit || 0),
        weeklyLossLimit: Number(risk?.weekly_loss_limit || 0),
        maxDrawdown: Number(risk?.max_drawdown || 0),
        maxPositions: Number(risk?.max_positions || 0),
        maxExposure: Number(risk?.max_exposure || 0),
        tradesPerDay: Number(risk?.trades_per_day || 0),
        tradesPerContext: Number(risk?.trades_per_context || 0),
        lossesPerBias: Number(risk?.losses_per_bias || 0),
    };
}

export default function SettingsPage() {
    const { token } = useAuth();
    const { switchMode } = useSession();

    const [settings, setSettings] = useState<SettingsState>(defaultSettings);
    const [riskDraft, setRiskDraft] = useState<RiskLimits>(defaultRiskLimits);
    const [mode, setMode] = useState<TradingMode>('testnet');
    const [loading, setLoading] = useState(true);
    const [savingRisk, setSavingRisk] = useState(false);
    const [showModeConfirm, setShowModeConfirm] = useState(false);
    const [showApiModal, setShowApiModal] = useState(false);
    const [targetMode, setTargetMode] = useState<TradingMode>('testnet');
    const [killSwitchStep, setKillSwitchStep] = useState(0);
    const [notice, setNotice] = useState<{ tone: NoticeTone; message: string } | null>(null);

    const fetchSettings = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/settings`, {
                headers: token ? { Authorization: `Bearer ${token}` } : {},
                cache: 'no-store',
            });

            if (!response.ok) {
                throw new Error('Failed to fetch settings');
            }

            const payload = await response.json();
            const data: any = unwrapApiData(payload);
            const nextRisk = mapRiskLimits(data.risk);
            const nextMode: TradingMode = data.trading_mode === 'live' ? 'live' : 'testnet';
            const nextConnectionStatus: ConnectionStatus =
                data.connection_status === 'connected' ||
                    data.connection_status === 'connecting' ||
                    data.connection_status === 'error'
                    ? data.connection_status
                    : 'disconnected';
            const nextEngineStatus: EngineStatus =
                data.engine_status === 'running' || data.engine_status === 'paused'
                    ? data.engine_status
                    : 'stopped';

            setSettings({
                tradingMode: nextMode,
                engineStatus: nextEngineStatus,
                apiConnection: {
                    exchange: data.exchange || 'Binance',
                    status: nextConnectionStatus,
                    lastPing: data.latency_ms ? `${data.latency_ms}ms` : '--',
                    testnet: data.testnet !== false,
                },
                riskLimits: nextRisk,
            });
            setRiskDraft(nextRisk);
            setMode(nextMode);
        } catch (error) {
            console.error('Failed to fetch settings:', error);
            setNotice({
                tone: 'error',
                message: error instanceof Error ? error.message : 'Failed to fetch settings',
            });
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        fetchSettings();
    }, [fetchSettings]);

    const handleModeChange = () => {
        const nextMode: TradingMode = mode === 'testnet' ? 'live' : 'testnet';
        setTargetMode(nextMode);
        if (nextMode === 'live') {
            setShowModeConfirm(true);
            return;
        }
        setShowApiModal(true);
    };

    const confirmModeChange = () => {
        setShowModeConfirm(false);
        setShowApiModal(true);
    };

    const handleApiSubmit = async (config: { apiKey: string; apiSecret: string; isValidated: boolean }) => {
        const success = await switchMode(targetMode, config);
        if (success) {
            setShowApiModal(false);
            setNotice({
                tone: 'success',
                message: `Trading mode switched to ${targetMode.toUpperCase()}.`,
            });
            window.dispatchEvent(new CustomEvent('accountSwitched'));
            await fetchSettings();
        } else {
            setNotice({
                tone: 'error',
                message: `Failed to switch to ${targetMode.toUpperCase()}.`,
            });
        }
        return success;
    };

    const handleRiskChange = (key: keyof RiskLimits, rawValue: string) => {
        const numericValue = Number(rawValue);
        setRiskDraft((current) => ({
            ...current,
            [key]: Number.isFinite(numericValue) ? numericValue : 0,
        }));
    };

    const handleRiskSave = async () => {
        setSavingRisk(true);
        setNotice(null);
        try {
            const response = await fetch(`${API_URL}/api/settings/risk`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({
                    daily_loss_limit: riskDraft.dailyLossLimit,
                    weekly_loss_limit: riskDraft.weeklyLossLimit,
                    max_drawdown: riskDraft.maxDrawdown,
                    max_positions: riskDraft.maxPositions,
                    max_exposure: riskDraft.maxExposure,
                    trades_per_day: riskDraft.tradesPerDay,
                    trades_per_context: riskDraft.tradesPerContext,
                    losses_per_bias: riskDraft.lossesPerBias,
                }),
            });

            if (!response.ok) {
                const payload = await response.json().catch(() => null);
                const detail = payload?.detail || payload?.message || 'Failed to save risk limits';
                throw new Error(detail);
            }

            const payload = await response.json();
            const data: any = unwrapApiData(payload);
            const nextRisk = mapRiskLimits(data.risk);

            setSettings((current) => ({
                ...current,
                riskLimits: nextRisk,
            }));
            setRiskDraft(nextRisk);
            setNotice({
                tone: 'success',
                message: 'Risk limits saved successfully.',
            });
        } catch (error) {
            setNotice({
                tone: 'error',
                message: error instanceof Error ? error.message : 'Failed to save risk limits',
            });
        } finally {
            setSavingRisk(false);
        }
    };

    const handleKillSwitch = async () => {
        if (killSwitchStep === 0) {
            setKillSwitchStep(1);
            return;
        }

        if (killSwitchStep === 1) {
            setKillSwitchStep(2);
            try {
                await fetch(`${API_URL}/api/kill-switch?active=true&reason=Manual%20activation%20from%20settings`, {
                    method: 'POST',
                    headers: token ? { Authorization: `Bearer ${token}` } : {},
                });
                window.dispatchEvent(new CustomEvent('accountSwitched'));
                setNotice({
                    tone: 'success',
                    message: 'Kill switch activated.',
                });
                await fetchSettings();
            } catch (error) {
                setNotice({
                    tone: 'error',
                    message: error instanceof Error ? error.message : 'Failed to activate kill switch',
                });
            }
            setTimeout(() => setKillSwitchStep(0), 5000);
        }
    };

    const riskChanged = JSON.stringify(riskDraft) !== JSON.stringify(settings.riskLimits);
    const statusBadgeClass =
        settings.apiConnection.status === 'connected'
            ? 'badge-success'
            : settings.apiConnection.status === 'connecting'
                ? 'badge-warning'
                : 'badge-danger';
    const engineBadgeClass =
        settings.engineStatus === 'running'
            ? 'badge-success'
            : settings.engineStatus === 'paused'
                ? 'badge-warning'
                : 'badge-neutral';

    if (loading) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Settings</h1>
                    <p className="text-[#8b98a5] text-sm">Safe system control</p>
                </div>
                <div className="card text-sm text-[#8b98a5]">Loading settings...</div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Settings</h1>
                <p className="text-[#8b98a5] text-sm">
                    Safe system control - dangerous actions require confirmation
                </p>
            </div>

            {notice && (
                <div
                    className={`rounded-md border px-4 py-3 text-sm ${notice.tone === 'success'
                            ? 'border-[#4a9268] bg-[#4a9268]/10 text-[#7fc395]'
                            : 'border-[#a65454] bg-[#a65454]/10 text-[#d28383]'
                        }`}
                >
                    {notice.message}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Trading Mode</span>
                        <span className={`badge ${engineBadgeClass}`}>
                            {settings.engineStatus.toUpperCase()}
                        </span>
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
                                {mode === 'testnet' ? 'TESTNET' : 'LIVE'}
                            </span>
                        </div>

                        <button
                            onClick={handleModeChange}
                            className={`btn w-full ${mode === 'testnet' ? 'btn-danger' : 'btn-ghost'}`}
                        >
                            {mode === 'testnet' ? 'Switch to LIVE Mode' : 'Switch to TESTNET Mode'}
                        </button>
                    </div>
                </div>

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
                            <span className={`badge ${statusBadgeClass}`}>
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

                <div className="card lg:col-span-2">
                    <div className="card-header">
                        <span className="card-title">Risk Limits</span>
                        <button
                            onClick={handleRiskSave}
                            disabled={!riskChanged || savingRisk}
                            className="btn btn-primary disabled:opacity-50"
                        >
                            {savingRisk ? 'Saving...' : 'Save Limits'}
                        </button>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {riskFieldMeta.map((field) => (
                            <div key={field.key} className="bg-[#1a1f26] rounded-md p-4 space-y-3">
                                <span className="label block">{field.label}</span>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="number"
                                        step={field.step ?? 1}
                                        min={field.min ?? 0}
                                        value={riskDraft[field.key]}
                                        onChange={(event) => handleRiskChange(field.key, event.target.value)}
                                        className="w-full bg-[#0f1419] border border-[#2d3640] rounded-md px-3 py-2 text-[#e7e9ea] focus:outline-none focus:border-[#5b7a9d]"
                                    />
                                    {field.suffix && (
                                        <span className="text-sm text-[#8b98a5]">{field.suffix}</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="card lg:col-span-2 border-[#a65454]">
                    <div className="card-header">
                        <span className="card-title text-[#a65454]">Emergency Kill Switch</span>
                    </div>

                    <div className="flex items-center justify-between gap-4">
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

            {showModeConfirm && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="card max-w-md mx-4">
                        <h3 className="heading-md mb-4">Confirm LIVE Mode</h3>
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

            <ApiKeyModal
                isOpen={showApiModal}
                targetMode={targetMode}
                onClose={() => setShowApiModal(false)}
                onSubmit={handleApiSubmit}
            />
        </div>
    );
}
