'use client';

import { useState, useEffect } from 'react';
import {
    LineChart,
    Line,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts';

interface PriceData {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export default function PriceChart() {
    const [priceData, setPriceData] = useState<PriceData[]>([]);
    const [timeframe, setTimeframe] = useState('1h');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Fetch price data from API
        const fetchPriceData = async () => {
            try {
                const response = await fetch(`/api/prices?timeframe=${timeframe}`);
                const data = await response.json();
                setPriceData(data);
                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch price data:', error);
                // Use mock data for demo
                const mockData: PriceData[] = Array.from({ length: 100 }, (_, i) => ({
                    timestamp: new Date(Date.now() - (99 - i) * 3600000).toISOString(),
                    open: 40000 + Math.random() * 5000,
                    high: 40500 + Math.random() * 5000,
                    low: 39500 + Math.random() * 5000,
                    close: 40000 + Math.random() * 5000,
                    volume: 1000 + Math.random() * 500,
                }));
                setPriceData(mockData);
                setLoading(false);
            }
        };

        fetchPriceData();
        const interval = setInterval(fetchPriceData, 30000); // Update every 30s
        return () => clearInterval(interval);
    }, [timeframe]);

    const currentPrice = priceData[priceData.length - 1]?.close || 0;
    const priceChange = priceData.length > 1
        ? ((currentPrice - priceData[0].close) / priceData[0].close) * 100
        : 0;

    return (
        <div className="card">
            {/* Header */}
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h2 className="text-xl font-bold">BTC/USDT</h2>
                    <div className="flex items-center gap-4 mt-1">
                        <span className="text-2xl font-mono">
                            ${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </span>
                        <span className={priceChange >= 0 ? 'status-long' : 'status-red'}>
                            {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                        </span>
                    </div>
                </div>

                {/* Timeframe Selector */}
                <div className="flex gap-2">
                    {['15m', '1h', '4h', '1d'].map((tf) => (
                        <button
                            key={tf}
                            onClick={() => setTimeframe(tf)}
                            className={`px-3 py-1 rounded text-sm ${timeframe === tf
                                    ? 'bg-accent-blue text-white'
                                    : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
                                }`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>
            </div>

            {/* Price Chart */}
            {loading ? (
                <div className="h-80 flex items-center justify-center">
                    <div className="text-text-secondary">Loading chart...</div>
                </div>
            ) : (
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={priceData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2e3338" />
                        <XAxis
                            dataKey="timestamp"
                            stroke="#5f6368"
                            tick={{ fill: '#9aa0a6' }}
                            tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                        />
                        <YAxis
                            stroke="#5f6368"
                            tick={{ fill: '#9aa0a6' }}
                            domain={['dataMin - 1000', 'dataMax + 1000']}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#16181d',
                                border: '1px solid #2e3338',
                                borderRadius: '8px',
                            }}
                            labelStyle={{ color: '#e8eaed' }}
                        />
                        <Line
                            type="monotone"
                            dataKey="close"
                            stroke="#2979ff"
                            strokeWidth={2}
                            dot={false}
                            animationDuration={300}
                        />
                    </LineChart>
                </ResponsiveContainer>
            )}

            {/* Volume Chart */}
            <ResponsiveContainer width="100%" height={100} className="mt-4">
                <BarChart data={priceData}>
                    <Bar dataKey="volume" fill="#5f6368" opacity={0.5} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
