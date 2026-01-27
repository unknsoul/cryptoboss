'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import { useState } from 'react';
import { Sidebar } from '../components/layout/Sidebar';
import { Topbar } from '../components/layout/Topbar';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper');
    const [systemStatus, setSystemStatus] = useState<'healthy' | 'warning' | 'critical' | 'unknown'>('healthy');

    const handleKillSwitch = () => {
        console.log('Kill switch activated');
        setSystemStatus('critical');
    };

    const handleModeToggle = () => {
        setTradingMode(tradingMode === 'paper' ? 'live' : 'paper');
    };

    return (
        <html lang="en" className="dark">
            <head>
                <title>CryptoBoss Dashboard</title>
                <meta name="description" content="Professional crypto trading control panel" />
            </head>
            <body className={`${inter.className} bg-[#0f1419] text-[#e7e9ea] antialiased`}>
                {/* Sidebar */}
                <Sidebar
                    collapsed={sidebarCollapsed}
                    onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
                />

                {/* Topbar - adjust position based on sidebar */}
                <div className={`transition-all duration-300 ${sidebarCollapsed ? 'pl-16' : 'pl-56'}`}>
                    <header className="fixed top-0 right-0 z-30 h-14 bg-[#0f1419] border-b border-[#2d3640]"
                        style={{ left: sidebarCollapsed ? '4rem' : '14rem' }}>
                        <div className="flex h-full items-center justify-between px-6">
                            {/* Left: System Status */}
                            <div className="flex items-center gap-4">
                                <div className="flex items-center gap-2">
                                    <div className={`w-2.5 h-2.5 rounded-full animate-pulse ${systemStatus === 'healthy' ? 'bg-green-500' :
                                            systemStatus === 'warning' ? 'bg-yellow-500' :
                                                systemStatus === 'critical' ? 'bg-red-500' : 'bg-gray-500'
                                        }`} />
                                    <span className="text-sm text-[#8b98a5]">
                                        {systemStatus === 'healthy' ? 'System Healthy' :
                                            systemStatus === 'warning' ? 'Degraded' :
                                                systemStatus === 'critical' ? 'Critical' : 'Unknown'}
                                    </span>
                                </div>
                            </div>

                            {/* Right: Controls */}
                            <div className="flex items-center gap-4">
                                {/* Last Decision */}
                                <div className="text-sm text-[#8b98a5]">
                                    No recent decisions
                                </div>

                                {/* Mode Toggle */}
                                <button
                                    onClick={handleModeToggle}
                                    className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${tradingMode === 'paper'
                                            ? 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
                                            : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                        }`}
                                >
                                    {tradingMode === 'paper' ? 'PAPER' : 'LIVE'}
                                </button>

                                {/* Kill Switch */}
                                <button
                                    onClick={handleKillSwitch}
                                    className="px-3 py-1.5 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700 transition-colors"
                                >
                                    KILL SWITCH
                                </button>
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
            </body>
        </html>
    );
}
