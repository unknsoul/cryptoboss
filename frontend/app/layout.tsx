'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from '../components/layout/Sidebar';
import { Topbar } from '@/components/layout/Topbar';
import { SessionProvider, useSession, TradingMode } from '../contexts/SessionContext';
import { AuthProvider, useAuth } from '../contexts/AuthContext';
import { ApiKeyModal } from '../components/ApiKeyModal';
import { unwrapApiData } from '@/lib/api';

const inter = Inter({ subsets: ['latin'] });

// Exchange health stages per spec
type ExchangeHealthStage = 'NORMAL' | 'DEGRADED' | 'CLOSE_ONLY' | 'HALTED';
type SystemStatus = 'healthy' | 'warning' | 'critical' | 'unknown';
type EngineStatus = 'running' | 'paused' | 'stopped';
type ApiConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function DashboardContent({ children }: { children: React.ReactNode }) {
    const { state, switchMode, resetSession } = useSession();
    const { token } = useAuth();

    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [systemStatus, setSystemStatus] = useState<SystemStatus>('unknown');
    const [exchangeHealth, setExchangeHealth] = useState<ExchangeHealthStage>('NORMAL');
    const [lastDecisionTime, setLastDecisionTime] = useState<string | null>(null);
    const [killSwitchStep, setKillSwitchStep] = useState(0);
    const [showApiModal, setShowApiModal] = useState(false);
    const [targetMode, setTargetMode] = useState<TradingMode>('testnet');
    const [mounted, setMounted] = useState(false);
    const [engineStatus, setEngineStatus] = useState<EngineStatus>('stopped');
    const [connectionStatus, setConnectionStatus] = useState<ApiConnectionStatus>('disconnected');
    const [backendMode, setBackendMode] = useState<TradingMode>('testnet');

    useEffect(() => {
        setMounted(true);
    }, []);

    const deriveSystemStatus = useCallback((system: any): SystemStatus => {
        if (system.kill_switch?.active || ['INCIDENT_FREEZE', 'HALTED'].includes(system.incident_state)) {
            return 'critical';
        }
        if (system.connection_status === 'error') {
            return 'critical';
        }
        if (system.engine_status === 'running' && system.connection_status === 'connected') {
            return 'healthy';
        }
        if (
            system.engine_status === 'paused' ||
            system.connection_status === 'connecting' ||
            system.connection_status === 'connected' ||
            system.api_validated
        ) {
            return 'warning';
        }
        return 'unknown';
    }, []);

    const deriveExchangeHealth = useCallback((system: any): ExchangeHealthStage => {
        if (system.kill_switch?.active || ['INCIDENT_FREEZE', 'HALTED'].includes(system.incident_state)) {
            return 'HALTED';
        }
        if (system.trading_paused) {
            return 'CLOSE_ONLY';
        }
        if (system.connection_status === 'connected') {
            return 'NORMAL';
        }
        if (system.connection_status === 'connecting' || system.api_validated) {
            return 'DEGRADED';
        }
        return 'DEGRADED';
    }, []);

    const refreshRuntime = useCallback(async () => {
        try {
            const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
            const [systemResponse, decisionsResponse] = await Promise.all([
                fetch(`${API_URL}/api/system`, { headers, cache: 'no-store' }),
                fetch(`${API_URL}/api/v11/decisions?limit=1`, { headers, cache: 'no-store' }),
            ]);

            if (systemResponse.ok) {
                const payload = await systemResponse.json();
                const system: any = unwrapApiData(payload);
                setEngineStatus((system.engine_status || 'stopped') as EngineStatus);
                setConnectionStatus((system.connection_status || 'disconnected') as ApiConnectionStatus);
                setBackendMode(system.mode === 'live' ? 'live' : 'testnet');
                setSystemStatus(deriveSystemStatus(system));
                setExchangeHealth(deriveExchangeHealth(system));
            } else {
                setSystemStatus('unknown');
                setExchangeHealth('DEGRADED');
                setEngineStatus('stopped');
                setConnectionStatus('error');
            }

            if (decisionsResponse.ok) {
                const payload = await decisionsResponse.json();
                const data: any = unwrapApiData(payload);
                const decisions = Array.isArray(data) ? data : (data.decisions || []);
                const latest = decisions[0];
                setLastDecisionTime(
                    latest?.timestamp
                        ? new Date(latest.timestamp).toLocaleTimeString('en-GB', {
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                        })
                        : null,
                );
            }
        } catch (error) {
            console.error('Failed to refresh runtime state:', error);
            setSystemStatus('unknown');
            setExchangeHealth('DEGRADED');
            setEngineStatus('stopped');
            setConnectionStatus('error');
        }
    }, [deriveExchangeHealth, deriveSystemStatus, token]);

    useEffect(() => {
        if (!mounted) {
            return;
        }
        refreshRuntime();
        const interval = setInterval(refreshRuntime, 5000);
        return () => clearInterval(interval);
    }, [mounted, refreshRuntime]);

    useEffect(() => {
        const handleAccountSwitched = () => {
            refreshRuntime();
        };
        window.addEventListener('accountSwitched', handleAccountSwitched);
        return () => window.removeEventListener('accountSwitched', handleAccountSwitched);
    }, [refreshRuntime]);

    const handleKillSwitch = async () => {
        try {
            setKillSwitchStep(1);
            await fetch(`${API_URL}/api/kill-switch?active=true&reason=Manual%20activation%20from%20topbar`, {
                method: 'POST',
                headers: token ? { Authorization: `Bearer ${token}` } : undefined,
            });
            setKillSwitchStep(2);
            await refreshRuntime();
        } catch (error) {
            console.error('Failed to activate kill switch:', error);
        } finally {
            setTimeout(() => setKillSwitchStep(0), 5000);
        }
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
            await refreshRuntime();
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
                engineStatus={engineStatus}
                connectionStatus={connectionStatus}
            />

            <Topbar
                sidebarCollapsed={sidebarCollapsed}
                systemStatus={systemStatus}
                tradingMode={backendMode}
                lastDecision={lastDecisionTime}
                onKillSwitch={handleKillSwitch}
                onModeToggle={() => handleModeChange(backendMode === 'live' ? 'testnet' : 'live')}
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
