'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import { useState, useEffect } from 'react';
import { Sidebar } from '../components/layout/Sidebar';
import { Topbar } from '@/components/layout/Topbar';
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

            <Topbar
                sidebarCollapsed={sidebarCollapsed}
                systemStatus={systemStatus}
                tradingMode={state.mode === 'live' ? 'live' : 'testnet'}
                lastDecision={lastDecisionTime}
                onKillSwitch={handleKillSwitch}
                onModeToggle={() => handleModeChange(state.mode === 'live' ? 'testnet' : 'live')}
            />

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
                <title>CryptoBoss v12.0 | Professional SMC Scalper Dashboard</title>
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
