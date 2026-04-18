'use client';

import { useState, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
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
    ReferenceArea,
    ReferenceLine,
} from 'recharts';

interface PriceData {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface OBOverlay {
    id: string;
    type: 'bullish' | 'bearish';
    status: string;
    top: number;
    bottom: number;
}

interface FVGOverlay {
    id: string;
    type: 'bullish' | 'bearish';
    status: string;
    top: number;
    bottom: number;
}

interface StructureOverlay {
    id: string;
    type: string;
    broken_level: number;
    direction: 'bullish' | 'bearish';
}

interface LiquidityOverlay {
    id: string;
    type: 'buyside' | 'sellside';
    status: string;
    price: number;
}

interface SMCOverlayState {
    order_blocks: OBOverlay[];
    fvgs: FVGOverlay[];
    structure: StructureOverlay[];
    liquidity: LiquidityOverlay[];
}

export default function PriceChart() {
    const [priceData, setPriceData] = useState<PriceData[]>([]);
    const [timeframe, setTimeframe] = useState('1h');
    const [loading, setLoading] = useState(true);
    const [overlays, setOverlays] = useState<SMCOverlayState>({
        order_blocks: [],
        fvgs: [],
        structure: [],
        liquidity: [],
    });

    useEffect(() => {
        const fetchPriceData = async () => {
            try {
                const [pricesResponse, smcResponse] = await Promise.all([
                    fetch(`${API_URL}/api/prices?timeframe=${timeframe}&limit=240`),
                    fetch(`${API_URL}/api/v2/smc/state?symbol=BTC%2FUSDT&timeframe=${timeframe}`),
                ]);

                const pricesJson = await pricesResponse.json();
                const pricesPayload = pricesJson.data || pricesJson;
                if (Array.isArray(pricesPayload)) {
                    setPriceData(pricesPayload as PriceData[]);
                } else {
                    setPriceData([]);
                }

                if (smcResponse.ok) {
                    const smcJson = await smcResponse.json();
                    const smcPayload = smcJson.data || smcJson;
                    const tfState = smcPayload.smc_state?.[timeframe] || {
                        order_blocks: [],
                        fvgs: [],
                        structure: [],
                        liquidity: [],
                    };
                    setOverlays({
                        order_blocks: tfState.order_blocks || [],
                        fvgs: tfState.fvgs || [],
                        structure: tfState.structure || [],
                        liquidity: tfState.liquidity || [],
                    });
                }

                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch price data:', error);
                setPriceData([]);
                setOverlays({ order_blocks: [], fvgs: [], structure: [], liquidity: [] });
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
    const xStart = priceData[0]?.timestamp;
    const xEnd = priceData[priceData.length - 1]?.timestamp;

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

                        {/* SMC Overlay Layer */}
                        {xStart && xEnd && overlays.order_blocks.slice(0, 4).map((ob) => (
                            <ReferenceArea
                                key={`ob-${ob.id}`}
                                x1={xStart}
                                x2={xEnd}
                                y1={ob.bottom}
                                y2={ob.top}
                                fill={ob.type === 'bullish' ? '#4a9268' : '#a65454'}
                                fillOpacity={0.08}
                                strokeOpacity={0}
                            />
                        ))}

                        {xStart && xEnd && overlays.fvgs.slice(0, 4).map((fvg) => (
                            <ReferenceArea
                                key={`fvg-${fvg.id}`}
                                x1={xStart}
                                x2={xEnd}
                                y1={fvg.bottom}
                                y2={fvg.top}
                                fill={fvg.type === 'bullish' ? '#5b7a9d' : '#c4a052'}
                                fillOpacity={0.07}
                                strokeOpacity={0}
                            />
                        ))}

                        {overlays.structure.slice(0, 5).map((structure) => (
                            <ReferenceLine
                                key={`structure-${structure.id}`}
                                y={structure.broken_level}
                                stroke={structure.direction === 'bullish' ? '#4a9268' : '#a65454'}
                                strokeDasharray="3 3"
                                strokeOpacity={0.8}
                            />
                        ))}

                        {overlays.liquidity.slice(0, 6).map((level) => (
                            <ReferenceLine
                                key={`liq-${level.id}`}
                                y={level.price}
                                stroke={level.type === 'buyside' ? '#a65454' : '#4a9268'}
                                strokeDasharray="2 5"
                                strokeOpacity={0.45}
                            />
                        ))}

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
