'use client';

import { useEffect, useState } from 'react';
import { PriceData, PriceSource, isStale, getPriceAge } from '@/hooks/usePriceValidation';

/**
 * Price Source Badge
 * 
 * RENDERING RULES:
 * - LIVE shown in neutral color
 * - TESTNET shown in warning color
 * - STALE shown in red
 * - REPLAY shown in purple
 */
interface PriceSourceBadgeProps {
    source: PriceSource;
    isStale?: boolean;
}

export function PriceSourceBadge({ source, isStale: priceIsStale }: PriceSourceBadgeProps) {
    const getSourceStyles = () => {
        if (priceIsStale) {
            return 'bg-red-500/20 text-red-300 border-red-500/30';
        }

        switch (source) {
            case 'LIVE_EXCHANGE_TICKER':
                return 'bg-green-500/20 text-green-300 border-green-500/30';
            case 'TESTNET_TICKER':
                return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
            case 'REPLAY_PRICE':
                return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
            case 'DERIVED_PRICE':
                return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
            default:
                return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
        }
    };

    const getLabel = () => {
        if (priceIsStale) return 'STALE';

        switch (source) {
            case 'LIVE_EXCHANGE_TICKER':
                return 'LIVE';
            case 'TESTNET_TICKER':
                return 'TESTNET';
            case 'REPLAY_PRICE':
                return 'REPLAY';
            case 'DERIVED_PRICE':
                return 'DERIVED';
            default:
                return 'UNKNOWN';
        }
    };

    return (
        <span className={`px-2 py-0.5 text-xs font-medium rounded border ${getSourceStyles()}`}>
            {getLabel()}
        </span>
    );
}


/**
 * Latency Indicator
 * 
 * Shows price age with color coding
 */
interface LatencyIndicatorProps {
    ageMs: number;
    maxAgeMs: number;
}

export function LatencyIndicator({ ageMs, maxAgeMs }: LatencyIndicatorProps) {
    const getColor = () => {
        const ratio = ageMs / maxAgeMs;
        if (ratio > 1) return 'text-red-400';
        if (ratio > 0.7) return 'text-yellow-400';
        return 'text-green-400';
    };

    return (
        <span className={`text-xs font-mono ${getColor()}`}>
            {ageMs < 1000 ? `${ageMs}ms` : `${(ageMs / 1000).toFixed(1)}s`}
        </span>
    );
}


/**
 * Price Display Component
 * 
 * REQUIRED ELEMENTS:
 * - Price Source Badge (LIVE / TESTNET / REPLAY)
 * - Last Updated Timestamp
 * - Exchange Name
 * - Latency Indicator
 * 
 * VISUAL RULES:
 * - TESTNET prices shown in warning color
 * - LIVE prices shown in neutral color
 * - STALE prices shown in red
 * - No animations for stale data
 */
interface PriceDisplayProps {
    price: PriceData | null;
    symbol: string;
    showDetails?: boolean;
    className?: string;
}

export function PriceDisplay({ price, symbol, showDetails = true, className = '' }: PriceDisplayProps) {
    const [now, setNow] = useState(Date.now());

    // Update "now" every second for age calculation
    useEffect(() => {
        const interval = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(interval);
    }, []);

    // No price available - show empty placeholder
    if (!price) {
        return (
            <div className={`flex flex-col ${className}`}>
                <div className="flex items-center gap-2">
                    <span className="text-2xl font-mono text-gray-500">---.--</span>
                    <PriceSourceBadge source="UNKNOWN" />
                </div>
                <span className="text-xs text-gray-500">{symbol} unavailable</span>
            </div>
        );
    }

    const priceIsStale = isStale(price);
    const ageMs = getPriceAge(price.timestamp_ms);

    // Max age for this source
    const maxAgeMs: Record<PriceSource, number> = {
        LIVE_EXCHANGE_TICKER: 2000,
        TESTNET_TICKER: 5000,
        DERIVED_PRICE: 10000,
        REPLAY_PRICE: 60000,
        UNKNOWN: 0,
    };

    const getPriceColor = () => {
        if (priceIsStale || !price.is_valid) return 'text-red-400';
        if (price.source === 'TESTNET_TICKER') return 'text-yellow-300';
        return 'text-white';
    };

    return (
        <div className={`flex flex-col ${className}`}>
            {/* Main Price Row */}
            <div className="flex items-center gap-3">
                {/* Price Value */}
                <span className={`text-2xl font-mono font-medium ${getPriceColor()} ${!priceIsStale && price.is_valid ? 'transition-all' : ''}`}>
                    ${price.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>

                {/* Source Badge */}
                <PriceSourceBadge source={price.source} isStale={priceIsStale} />

                {/* Latency */}
                <LatencyIndicator ageMs={ageMs} maxAgeMs={maxAgeMs[price.source]} />
            </div>

            {/* Details Row */}
            {showDetails && (
                <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
                    <span>{price.symbol}</span>
                    <span>•</span>
                    <span>{price.exchange}</span>
                    <span>•</span>
                    <span>{new Date(price.timestamp_ms).toLocaleTimeString()}</span>
                    {price.bid && price.ask && (
                        <>
                            <span>•</span>
                            <span>Spread: {((price.ask - price.bid) / price.price * 100).toFixed(3)}%</span>
                        </>
                    )}
                </div>
            )}

            {/* Stale Warning */}
            {priceIsStale && (
                <div className="text-xs text-red-400 mt-1 flex items-center gap-1">
                    <span>⚠️</span>
                    <span>Price data is stale ({(ageMs / 1000).toFixed(1)}s old)</span>
                </div>
            )}

            {/* Invalid Warning */}
            {!price.is_valid && price.rejection_reason && (
                <div className="text-xs text-red-400 mt-1 flex items-center gap-1">
                    <span>❌</span>
                    <span>{price.rejection_reason}</span>
                </div>
            )}
        </div>
    );
}


/**
 * Compact Price Display
 * 
 * For use in tables and tight spaces
 */
interface CompactPriceProps {
    price: PriceData | null;
    showSource?: boolean;
}

export function CompactPrice({ price, showSource = true }: CompactPriceProps) {
    if (!price) {
        return <span className="text-gray-500 font-mono">---.--</span>;
    }

    const priceIsStale = isStale(price);
    const getPriceColor = () => {
        if (priceIsStale || !price.is_valid) return 'text-red-400';
        if (price.source === 'TESTNET_TICKER') return 'text-yellow-300';
        return 'text-white';
    };

    return (
        <span className="flex items-center gap-1.5">
            <span className={`font-mono ${getPriceColor()}`}>
                ${price.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            {showSource && (
                <span className={`text-[10px] px-1 rounded ${priceIsStale ? 'bg-red-500/20 text-red-300' :
                        price.source === 'LIVE_EXCHANGE_TICKER' ? 'bg-green-500/20 text-green-300' :
                            price.source === 'TESTNET_TICKER' ? 'bg-yellow-500/20 text-yellow-300' :
                                'bg-gray-500/20 text-gray-400'
                    }`}>
                    {priceIsStale ? 'STALE' :
                        price.source === 'LIVE_EXCHANGE_TICKER' ? 'L' :
                            price.source === 'TESTNET_TICKER' ? 'T' : '?'}
                </span>
            )}
        </span>
    );
}
