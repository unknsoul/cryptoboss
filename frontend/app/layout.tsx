'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import { useState, useEffect } from 'react';
import { Sidebar } from '../components/layout/Sidebar';
import { SessionProvider, useSession, TradingMode } from '../contexts/SessionContext';
import { AuthProvider, useAuth } from '../contexts/AuthContext';
import { ApiKeyModal } from '../components/ApiKeyModal';

const inter = Inter({ subsets: ['latin'] });

// Exchange health stages per spec
type ExchangeHealthStage = 'NORMAL' | 'DEGRADED' | 'CLOSE_ONLY' | 'HALTED';
type SystemStatus = 'healthy' | 'warning' | 'critical' | 'unknown';

function DashboardContent({ children }: { children: React.ReactNode }) {
    const { state, switchMode, resetSession } = useSession();
    const { user, activeAccount, isAuthenticated, logout } = useAuth();

    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [systemStatus, setSystemStatus] = useState<SystemStatus>('healthy');
    const [exchangeHealth, setExchangeHealth] = useState<ExchangeHealthStage>('NORMAL');
    const [lastDecisionTime, setLastDecisionTime] = useState<string | null>(null);
    const [killSwitchStep, setKillSwitchStep] = useState(0);
    const [showApiModal, setShowApiModal] = useState(false);
    const [targetMode, setTargetMode] = useState<TradingMode>('testnet');
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    const handleKillSwitch = () => {
        if (killSwitchStep === 0) {
            setKillSwitchStep(1);
            setTimeout(() => setKillSwitchStep(0), 3000);
            return;
        }
        // Second click - activate
        console.log('KILL SWITCH ACTIVATED');
        setSystemStatus('critical');
        setExchangeHealth('HALTED');
        setKillSwitchStep(2);
        setTimeout(() => setKillSwitchStep(0), 5000);
    };

    const handleModeChange = (newMode: TradingMode) => {
        // PAPER REMOVED - always require API keys for testnet/live
        setTargetMode(newMode);
        setShowApiModal(true);
    };

    const handleApiSubmit = async (config: { apiKey: string; apiSecret: string; isValidated: boolean }) => {
        const success = await switchMode(targetMode, config);
        if (success) {
            setShowApiModal(false);
        }
        return success;
    };

    const handleResetSession = () => {
        if (confirm('Reset session? This will clear all cached data.')) {
            resetSession();
        }
    };

    const getStatusColor = (status: SystemStatus) => {
        switch (status) {
            case 'healthy': return 'bg-[#4a9268]';
            case 'warning': return 'bg-[#c4a052]';
            case 'critical': return 'bg-[#a65454]';
            default: return 'bg-[#6b7280]';
        }
    };

    const getStatusText = (status: SystemStatus) => {
        switch (status) {
            case 'healthy': return 'System Healthy';
            case 'warning': return 'Degraded';
            case 'critical': return 'Critical';
            default: return 'Unknown';
        }
    };

    const getExchangeHealthColor = (stage: ExchangeHealthStage) => {
        switch (stage) {
            case 'NORMAL': return 'badge-success';
            case 'DEGRADED': return 'badge-warning';
            case 'CLOSE_ONLY': return 'badge-warning';
            case 'HALTED': return 'badge-danger';
        }
    };

    const getModeColor = (mode: TradingMode) => {
        switch (mode) {
            case 'testnet': return 'bg-[rgba(196,160,82,0.15)] text-[#c4a052]';
            case 'live': return 'bg-[rgba(166,84,84,0.15)] text-[#a65454]';
            default: return 'bg-[rgba(196,160,82,0.15)] text-[#c4a052]'; // Default to testnet
        }
    };

    const getModeLabel = (mode: TradingMode) => {
        switch (mode) {
            case 'testnet': return '🟡 TESTNET';
            case 'live': return '🔴 LIVE';
            default: return '🟡 TESTNET';
        }
    };

    const getConnectionIcon = () => {
        switch (state.connectionStatus) {
            case 'connected': return '🟢';
            case 'connecting': return '🟡';
            case 'error': return '🔴';
            default: return '⚪';
        }
    };

    // Truncate session ID for display
    const shortSessionId = state.sessionId ? state.sessionId.slice(0, 12) + '...' : '--';

    return (
        <>
            {/* Sidebar */}
            <Sidebar
                collapsed={sidebarCollapsed}
                onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
            />

            {/* Topbar */}
            <div className={`transition-all duration-300 ${sidebarCollapsed ? 'pl-16' : 'pl-56'}`}>
                <header
                    className="fixed top-0 right-0 z-30 h-14 bg-[#0f1419] border-b border-[#2d3640]"
                    style={{ left: sidebarCollapsed ? '4rem' : '14rem' }}
                >
                    <div className="flex h-full items-center justify-between px-4">
                        {/* Left: System Status + Session Info */}
                        <div className="flex items-center gap-4">
                            {/* System State Indicator */}
                            <div className="flex items-center gap-2">
                                <div className={`status-dot ${getStatusColor(systemStatus)}`} />
                                <span className="text-sm text-[#8b98a5]">
                                    {getStatusText(systemStatus)}
                                </span>
                            </div>

                            {/* Exchange Health Stage */}
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-[#6b7280]">Exchange:</span>
                                <span className={`badge ${getExchangeHealthColor(exchangeHealth)}`}>
                                    {exchangeHealth}
                                </span>
                            </div>

                            {/* Connection Status */}
                            <div className="flex items-center gap-1">
                                <span>{getConnectionIcon()}</span>
                                <span className="text-xs text-[#6b7280] capitalize">
                                    {state.connectionStatus}
                                </span>
                            </div>

                            {/* Session ID */}
                            {mounted && (
                                <div className="flex items-center gap-2 text-xs">
                                    <span className="text-[#6b7280]">Session:</span>
                                    <code className="text-[#5b7a9d] font-mono">{shortSessionId}</code>
                                    <button
                                        onClick={handleResetSession}
                                        className="text-[#6b7280] hover:text-[#e7e9ea] text-xs"
                                        title="Reset Session"
                                    >
                                        🔄
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Right: Mode Selector + Kill Switch */}
                        <div className="flex items-center gap-3">
                            {/* Mode Selector - PAPER REMOVED */}
                            <div className="flex items-center gap-1 bg-[#1a1f26] rounded-md p-1">
                                <button
                                    onClick={() => handleModeChange('testnet')}
                                    className={`px-2 py-1 rounded text-xs font-medium transition-colors ${state.mode === 'testnet' ? getModeColor('testnet') : 'text-[#6b7280] hover:text-[#e7e9ea]'
                                        }`}
                                >
                                    Testnet
                                </button>
                                <button
                                    onClick={() => handleModeChange('live')}
                                    className={`px-2 py-1 rounded text-xs font-medium transition-colors ${state.mode === 'live' ? getModeColor('live') : 'text-[#6b7280] hover:text-[#e7e9ea]'
                                        }`}
                                >
                                    Live
                                </button>
                            </div>

                            {/* Current Mode Badge */}
                            <span className={`badge ${getModeColor(state.mode)}`}>
                                {getModeLabel(state.mode)}
                            </span>

                            {/* Kill Switch - Double confirmation per spec */}
                            <button
                                onClick={handleKillSwitch}
                                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${killSwitchStep === 2
                                    ? 'bg-[#6b7280] text-white cursor-not-allowed'
                                    : killSwitchStep === 1
                                        ? 'bg-[#c44444] text-white animate-pulse'
                                        : 'bg-[#a65454] text-white hover:bg-[#b66464]'
                                    }`}
                                disabled={killSwitchStep === 2}
                            >
                                {killSwitchStep === 0 && 'KILL'}
                                {killSwitchStep === 1 && 'CONFIRM'}
                                {killSwitchStep === 2 && 'HALTED'}
                            </button>

                            {/* v1.0.1: User & Account Indicator */}
                            {isAuthenticated && user && (
                                <div className="flex items-center gap-2 pl-3 border-l border-[#2d3640]">
                                    {activeAccount && (
                                        <span className={`badge text-xs ${activeAccount.environment === 'LIVE'
                                            ? 'bg-red-500/20 text-red-300'
                                            : 'bg-yellow-500/20 text-yellow-300'
                                            }`}>
                                            {activeAccount.environment}
                                        </span>
                                    )}
                                    <a
                                        href="/accounts"
                                        className="text-xs text-[#8b98a5] hover:text-[#e7e9ea] transition"
                                    >
                                        {user.email.split('@')[0]}
                                    </a>
                                    <button
                                        onClick={logout}
                                        className="text-xs text-[#6b7280] hover:text-[#a65454] transition"
                                        title="Logout"
                                    >
                                        ⏻
                                    </button>
                                </div>
                            )}
                            {!isAuthenticated && (
                                <a
                                    href="/auth/login"
                                    className="text-xs text-[#5b7a9d] hover:text-[#e7e9ea] transition"
                                >
                                    Sign In
                                </a>
                            )}
                        </div>
                    </div>
                </header>
            </div>

            {/* Main Content */}
            <main className={`pt-14 min-h-screen transition-all duration-300 ${sidebarCollapsed ? 'pl-16' : 'pl-56'
                }`}>
                <div className="p-6">
                    {children}
                </div>
            </main>

            {/* API Key Modal */}
            <ApiKeyModal
                isOpen={showApiModal}
                targetMode={targetMode}
                onClose={() => setShowApiModal(false)}
                onSubmit={handleApiSubmit}
            />
        </>
    );
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <head>
                <title>CryptoBoss v1.0.1 | Trading Control Dashboard</title>
                <meta name="description" content="Professional crypto trading control panel" />
            </head>
            <body className={`${inter.className} bg-[#0f1419] text-[#e7e9ea] antialiased`}>
                <AuthProvider>
                    <SessionProvider>
                        <DashboardContent>{children}</DashboardContent>
                    </SessionProvider>
                </AuthProvider>
            </body>
        </html>
    );
}
