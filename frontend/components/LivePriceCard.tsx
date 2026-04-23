'use client';

/**
 * LivePriceCard - Real-time price display using WebSocket
 * 
 * CRYPTOBOSS 2.0: All prices from backend WebSocket
 * - No polling, no fake data
 * - Shows "Waiting for exchange" when no data
 * - Shows "Disconnected" when socket is down
 */

import { usePriceSocket, PriceData, PriceSocketStatus, formatPrice } from '../hooks/usePriceSocket';
import { useAuth } from '../contexts/AuthContext';

interface LivePriceCardProps {
    symbol: string;
    showDetails?: boolean;
    className?: string;
}

// Get symbol display name
function getSymbolName(symbol: string): string {
    const names: Record<string, string> = {
        'BTCUSDT': 'Bitcoin',
        'ETHUSDT': 'Ethereum',
        'BNBUSDT': 'BNB',
        'SOLUSDT': 'Solana',
    };
    return names[symbol] || symbol;
}

// Get symbol icon
function getSymbolIcon(symbol: string): string {
    const icons: Record<string, string> = {
        'BTCUSDT': '₿',
        'ETHUSDT': 'Ξ',
        'BNBUSDT': '◈',
        'SOLUSDT': '◎',
    };
    return icons[symbol] || '◇';
}

export function LivePriceCard({ symbol, showDetails = true, className = '' }: LivePriceCardProps) {
    const { activeAccount } = useAuth();
    const { prices, status, isConnected, lastUpdate } = usePriceSocket({
        exchangeAccountId: activeAccount?.exchange_account_id,
        symbols: [symbol]
    });

    const priceData = prices[symbol];

    // Get status indicator
    const getStatusIndicator = () => {
        switch (status) {
            case 'connected':
                return <span className="status-dot status-dot-healthy" title="Connected" />;
            case 'connecting':
                return <span className="status-dot status-dot-warning animate-pulse" title="Connecting..." />;
            case 'disconnected':
            case 'error':
                return <span className="status-dot status-dot-critical" title="Disconnected" />;
        }
    };

    // Get price display
    const getPriceDisplay = () => {
        if (!isConnected && status !== 'connecting') {
            return (
                <span className="text-[#8b98a5]">Disconnected</span>
            );
        }

        if (!priceData) {
            return (
                <span className="text-[#8b98a5]">Waiting for exchange...</span>
            );
        }

        return (
            <span className="text-[#e7e9ea] font-mono">
                ${formatPrice(priceData.price, symbol.includes('BTC') ? 2 : 2)}
            </span>
        );
    };

    // Get source badge
    const getSourceBadge = () => {
        if (!priceData) return null;

        const source = priceData.source;
        const badgeClass = source === 'LIVE'
            ? 'badge-success'
            : source === 'TESTNET'
                ? 'badge-warning'
                : 'badge-neutral';

        return (
            <span className={`badge ${badgeClass} text-xs`}>
                {source}
            </span>
        );
    };

    // Get time ago
    const getTimeAgo = () => {
        if (!priceData?.timestamp) return null;

        const ageMs = Date.now() - priceData.timestamp;
        if (ageMs < 1000) return 'just now';
        if (ageMs < 60000) return `${Math.floor(ageMs / 1000)}s ago`;
        return `${Math.floor(ageMs / 60000)}m ago`;
    };

    return (
        <div className={`card ${className}`}>
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <span className="text-xl">{getSymbolIcon(symbol)}</span>
                    <div>
                        <span className="text-[#e7e9ea] font-medium">{getSymbolName(symbol)}</span>
                        <span className="text-[#6b7280] text-sm ml-2">{symbol}</span>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {getSourceBadge()}
                    {getStatusIndicator()}
                </div>
            </div>

            <div className="value-xl mb-2">
                {getPriceDisplay()}
            </div>

            {showDetails && priceData && (
                <div className="flex items-center justify-between text-xs text-[#6b7280]">
                    <span>Updated {getTimeAgo()}</span>
                    <span>Account: {activeAccount?.label || 'default'}</span>
                </div>
            )}

        </div>
    );
}

/**
 * LivePricesGrid - Grid of all price cards
 */
const DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'];

interface LivePricesGridProps {
    symbols?: string[];
    className?: string;
}

export function LivePricesGrid({ symbols = DEFAULT_SYMBOLS, className = '' }: LivePricesGridProps) {
    return (
        <div className={`grid grid-cols-2 lg:grid-cols-4 gap-4 ${className}`}>
            {symbols.map(symbol => (
                <LivePriceCard key={symbol} symbol={symbol} showDetails={false} />
            ))}
        </div>
    );
}

/**
 * PriceStatusBanner - Shows overall connection status
 */
interface PriceStatusBannerProps {
    className?: string;
}

export function PriceStatusBanner({ className = '' }: PriceStatusBannerProps) {
    const { activeAccount } = useAuth();
    const { status, error, isConnected, lastUpdate } = usePriceSocket({
        exchangeAccountId: activeAccount?.exchange_account_id
    });

    if (isConnected) {
        return null; // Don't show when connected
    }

    const getMessage = () => {
        if (!activeAccount) {
            return 'Select an exchange account to view live prices';
        }

        switch (status) {
            case 'connecting':
                return 'Connecting to exchange...';
            case 'error':
                return error || 'Connection error';
            case 'disconnected':
                return 'Disconnected from exchange';
            default:
                return 'Waiting for exchange data...';
        }
    };

    const getStatusClass = () => {
        switch (status) {
            case 'connecting':
                return 'bg-[#c4a052]/20 border-[#c4a052]';
            case 'error':
                return 'bg-[#a65454]/20 border-[#a65454]';
            default:
                return 'bg-[#5b7a9d]/20 border-[#5b7a9d]';
        }
    };

    return (
        <div className={`p-3 rounded-md border ${getStatusClass()} ${className}`}>
            <div className="flex items-center gap-2">
                {status === 'connecting' && (
                    <span className="animate-spin">⏳</span>
                )}
                {status === 'error' && <span>❌</span>}
                {status === 'disconnected' && <span>🔌</span>}
                <span className="text-sm">{getMessage()}</span>
            </div>
        </div>
    );
}

export default LivePriceCard;
