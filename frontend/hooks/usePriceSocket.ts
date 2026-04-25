'use client';

/**
 * usePriceSocket - Real-time price updates via WebSocket
 * 
 * CRYPTOBOSS 2.0: All prices come from backend WebSocket
 * - No polling
 * - No fallback/fake prices
 * - Shows "Waiting" / "Disconnected" states
 * 
 * Usage:
 *   const { prices, status, error } = usePriceSocket({
 *     exchangeAccountId: 'abc123',
 *     symbols: ['BTCUSDT', 'ETHUSDT']
 *   });
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';

export type PriceSocketStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface PriceData {
    symbol: string;
    price: number;
    timestamp: number;
    source: string;  // 'TESTNET', 'LIVE', 'BINANCE_MAINNET', etc.
}

export interface PricesState {
    [symbol: string]: PriceData;
}

export interface UsePriceSocketOptions {
    exchangeAccountId?: string;
    symbols?: string[];
    autoConnect?: boolean;
}

export interface UsePriceSocketReturn {
    prices: PricesState;
    status: PriceSocketStatus;
    error: string | null;
    connect: () => void;
    disconnect: () => void;
    isConnected: boolean;
    lastUpdate: number | null;
}

const DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'];

function getWsBase(): string {
    if (process.env.NEXT_PUBLIC_WS_URL) {
        return process.env.NEXT_PUBLIC_WS_URL;
    }

    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    return apiBase
        .replace(/^http/, 'ws')
        .replace('://localhost', '://127.0.0.1');
}

const WS_BASE = getWsBase();

export function usePriceSocket(options: UsePriceSocketOptions = {}): UsePriceSocketReturn {
    const { exchangeAccountId, autoConnect = true } = options;
    const symbolsKey = (options.symbols || DEFAULT_SYMBOLS).join(',');
    const { activeAccount } = useAuth();

    const [prices, setPrices] = useState<PricesState>({});
    const [status, setStatus] = useState<PriceSocketStatus>('disconnected');
    const [error, setError] = useState<string | null>(null);
    const [lastUpdate, setLastUpdate] = useState<number | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const maxReconnectAttempts = 5;

    // Use provided accountId or fall back to active account
    const accountId = exchangeAccountId || activeAccount?.exchange_account_id;

    const clearPrices = useCallback(() => {
        setPrices({});
        setLastUpdate(null);
    }, []);

    const connect = useCallback(() => {
        // Use default account if none selected
        const connectId = accountId || 'default';

        // Don't reconnect if already connecting/connected
        if (wsRef.current?.readyState === WebSocket.CONNECTING ||
            wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setStatus('connecting');
        setError(null);

        try {
            const wsUrl = `${WS_BASE}/ws/prices?account=${connectId}&symbols=${symbolsKey}`;

            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                setStatus('connected');
                setError(null);
                reconnectAttemptsRef.current = 0;
                console.log(`[PriceSocket] Connected for account ${accountId}`);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // Helper: parse timestamp to epoch ms
                    const parseTimestamp = (ts: any): number => {
                        if (!ts) return Date.now();
                        if (typeof ts === 'number') {
                            // If it's seconds (< 1e12), convert to ms
                            return ts < 1e12 ? ts * 1000 : ts;
                        }
                        // ISO string
                        const parsed = new Date(ts).getTime();
                        return isNaN(parsed) ? Date.now() : parsed;
                    };

                    // Handle individual price updates
                    if (data.type === 'price' || data.channel === 'prices') {
                        const priceData: PriceData = {
                            symbol: data.symbol || data.data?.symbol,
                            price: parseFloat(data.price ?? data.data?.price ?? 0),
                            timestamp: parseTimestamp(data.timestamp || data.data?.timestamp),
                            source: data.source || data.data?.source || 'TESTNET'
                        };

                        // Only update if we got a valid price
                        if (priceData.symbol && priceData.price > 0) {
                            // Only update if matches our account (or no account filter)
                            if (!data.exchange_account_id || data.exchange_account_id === accountId) {
                                setPrices(prev => ({
                                    ...prev,
                                    [priceData.symbol]: priceData
                                }));
                                setLastUpdate(Date.now());
                            }
                        }
                    }

                    // Handle heartbeat
                    if (data.type === 'heartbeat') {
                        // Keep connection alive
                    }

                } catch (e) {
                    console.error('[PriceSocket] Parse error:', e);
                }
            };

            ws.onerror = (event) => {
                console.error('[PriceSocket] Error:', event);
                setError('WebSocket connection error');
                setStatus('error');
            };

            ws.onclose = (event) => {
                setStatus('disconnected');
                wsRef.current = null;

                // Attempt reconnect if not intentionally closed
                if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
                    reconnectAttemptsRef.current++;
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
                    console.log(`[PriceSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, delay);
                } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
                    setError('Failed to connect after multiple attempts');
                    setStatus('error');
                }
            };

        } catch (e) {
            setError(e instanceof Error ? e.message : 'Connection failed');
            setStatus('error');
        }
    }, [accountId, symbolsKey]);

    const disconnect = useCallback(() => {
        // Clear reconnect timeout
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        // Close WebSocket
        if (wsRef.current) {
            wsRef.current.close(1000, 'User disconnected');
            wsRef.current = null;
        }

        setStatus('disconnected');
        reconnectAttemptsRef.current = 0;
    }, []);

    // Auto-connect when account changes
    useEffect(() => {
        if (autoConnect) {
            // Disconnect previous connection
            disconnect();
            // Clear prices for new account
            clearPrices();
            // Connect
            connect();
        }

        return () => {
            disconnect();
        };
    }, [accountId, autoConnect, connect, disconnect, clearPrices]);

    return {
        prices,
        status,
        error,
        connect,
        disconnect,
        isConnected: status === 'connected',
        lastUpdate
    };
}

/**
 * Get formatted price display
 */
export function formatPrice(price: number | undefined, decimals: number = 2): string {
    if (price === undefined || price === null) {
        return '--';
    }
    return price.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/**
 * Get price change indicator
 */
export function getPriceStatusClass(status: PriceSocketStatus): string {
    switch (status) {
        case 'connected':
            return 'status-dot-healthy';
        case 'connecting':
            return 'status-dot-warning';
        case 'disconnected':
        case 'error':
            return 'status-dot-critical';
        default:
            return '';
    }
}
