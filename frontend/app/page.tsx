'use client';

import { useState, useEffect } from 'react';
import PriceChart from '@/components/PriceChart';
import PositionsTable from '@/components/PositionsTable';
import RiskMetrics from '@/components/RiskMetrics';
import StrategyControls from '@/components/StrategyControls';
import ModeSwitch from '@/components/ModeSwitch';

export default function Dashboard() {
    const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper');
    const [equity, setEquity] = useState(10000);

    return (
        <div className="min-h-screen p-6">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-3xl font-bold">CryptoBoss Pro</h1>
                    <p className="text-text-secondary">Advanced Trading System</p>
                </div>

                <div className="flex items-center gap-4">
                    <div className="card">
                        <div className="text-text-secondary text-sm">Portfolio Value</div>
                        <div className="text-2xl font-bold">
                            ${equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </div>
                    </div>

                    <ModeSwitch mode={tradingMode} onModeChange={setTradingMode} />
                </div>
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Price Chart - Takes 2 columns */}
                <div className="lg:col-span-2">
                    <PriceChart />
                </div>

                {/* Risk Metrics - Sidebar */}
                <div>
                    <RiskMetrics />
                </div>

                {/* Positions Table - Full width */}
                <div className="lg:col-span-2">
                    <PositionsTable />
                </div>

                {/* Strategy Controls - Sidebar */}
                <div>
                    <StrategyControls />
                </div>
            </div>
        </div>
    );
}
