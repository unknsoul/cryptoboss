'use client';

/**
 * Live Status Page - CRYPTOBOSS vFINAL
 * 
 * SIMPLIFIED: Direct REST polling for reliability
 * Shows real prices from Binance
 */

import { useState, useEffect, useCallback } from 'react';

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'];
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_URL = `${API_BASE}/api/prices/live`;
const POLL_INTERVAL = 3000;

interface Price {
    symbol: string;
    price: number;
    change24h: number;
    high24h?: number;
    low24h?: number;
    volume24h?: number;
    timestamp: string;
    source?: string;
}

interface Prices {
    [key: string]: Price;
}

function PriceCard({ symbol, price }: { symbol: string; price: Price | null }) {
    const hasPrice = price && price.price > 0;
    const isPositive = (price?.change24h || 0) >= 0;

    return (
        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
            <div className="flex justify-between items-start mb-2">
                <span className="text-lg font-semibold text-white">
                    {symbol.replace('USDT', '/USDT')}
                </span>
                <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${hasPrice ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                        }`} />
                    <span className="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">
                        LIVE
                    </span>
                </div>
            </div>

            {!hasPrice ? (
                <div className="py-4">
                    <div className="text-3xl font-mono text-gray-500">---.--</div>
                    <div className="text-sm text-yellow-400 mt-1">Loading...</div>
                </div>
            ) : (
                <>
                    <div className="py-2">
                        <span className="text-3xl font-mono text-white">
                            ${price.price.toLocaleString('en-US', {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: price.price < 100 ? 4 : 2
                            })}
                        </span>
                    </div>

                    <div className="flex items-center gap-3 mt-2">
                        <span className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'
                            }`}>
                            {isPositive ? '+' : ''}{price.change24h?.toFixed(2) || '0.00'}%
                        </span>

                        {price.volume24h && (
                            <span className="text-xs text-gray-400">
                                Vol: ${(price.volume24h / 1000000).toFixed(1)}M
                            </span>
                        )}
                    </div>

                    {price.high24h && price.low24h && (
                        <div className="flex justify-between text-xs text-gray-500 mt-3 pt-3 border-t border-gray-700">
                            <span>H: ${price.high24h.toLocaleString()}</span>
                            <span>L: ${price.low24h.toLocaleString()}</span>
                        </div>
                    )}
                </>
            )}

            <div className="text-xs text-gray-600 mt-3">
                {price?.timestamp ? new Date(price.timestamp).toLocaleTimeString() : '--:--:--'}
            </div>
        </div>
    );
}

export default function LiveStatusPage() {
    const [prices, setPrices] = useState<Prices>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [lastUpdate, setLastUpdate] = useState<string>('--:--:--');

    const fetchPrices = useCallback(async () => {
        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            console.log('Price data:', result);

            if (result.data?.prices) {
                setPrices(result.data.prices);
                setLastUpdate(new Date().toLocaleTimeString('en-GB', { hour12: false }));
                setError(null);
            } else {
                setError('No price data received');
            }
        } catch (e: any) {
            console.error('Fetch error:', e);
            setError(e.message || 'Failed to fetch prices');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        // Initial fetch
        fetchPrices();

        // Poll every 3 seconds
        const interval = setInterval(fetchPrices, POLL_INTERVAL);

        return () => clearInterval(interval);
    }, [fetchPrices]);

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="mb-8">
                <div className="flex items-center gap-3 mb-1">
                    <h1 className="text-3xl font-bold text-white">Live Prices</h1>
                    <span className={`px-3 py-1 rounded text-sm font-medium ${error ? 'bg-red-500/20 text-red-400' :
                            loading ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-green-500/20 text-green-400'
                        }`}>
                        {error ? 'ERROR' : loading ? 'LOADING' : 'CONNECTED'}
                    </span>
                </div>
                <p className="text-gray-400 text-sm">
                    Real-time prices from Binance — Last update: {lastUpdate}
                </p>
                {error && (
                    <p className="text-red-400 text-sm mt-2">Error: {error}</p>
                )}
            </div>

            {/* Price Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {SYMBOLS.map(symbol => (
                    <PriceCard
                        key={symbol}
                        symbol={symbol}
                        price={prices[symbol] || null}
                    />
                ))}
            </div>

            {/* Info */}
            <div className="mt-8 p-4 bg-gray-800/50 rounded-xl border border-gray-700">
                <div className="flex items-start gap-4 text-sm text-gray-400">
                    <span className="text-xl">ℹ️</span>
                    <div>
                        <p className="mb-2">
                            <strong className="text-white">Real prices from Binance.</strong> Updates every 3 seconds.
                        </p>
                        <p>
                            Testnet trading uses mainnet prices (testnet has no real market data).
                        </p>
                    </div>
                </div>
            </div>

            {/* Debug Info */}
            <div className="mt-4 p-4 bg-gray-900 rounded-xl border border-gray-800 text-xs font-mono text-gray-500">
                <div>Symbols: {Object.keys(prices).join(', ') || 'None'}</div>
                <div>BTC: {prices['BTCUSDT']?.price || 'N/A'}</div>
            </div>
        </div>
    );
}
