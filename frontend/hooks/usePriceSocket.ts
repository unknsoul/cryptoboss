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
    source: 'TESTNET' | 'LIVE';
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

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export function usePriceSocket(options: UsePriceSocketOptions = {}): UsePriceSocketReturn {
    const { exchangeAccountId, symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'], autoConnect = true } = options;
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
        // Don't connect without an account
        if (!accountId) {
            setStatus('disconnected');
            setError('No exchange account selected');
            return;
        }

        // Don't reconnect if already connecting/connected
        if (wsRef.current?.readyState === WebSocket.CONNECTING ||
            wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setStatus('connecting');
        setError(null);

        try {
            const symbolsParam = symbols.join(',');
            const wsUrl = `${WS_BASE}/ws/prices?account=${accountId}&symbols=${symbolsParam}`;

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

                    // Handle price updates
                    if (data.type === 'price' || data.channel === 'prices') {
                        const priceData: PriceData = {
                            symbol: data.symbol || data.data?.symbol,
                            price: parseFloat(data.price || data.data?.price),
                            timestamp: data.timestamp || data.data?.timestamp || Date.now(),
                            source: data.source || data.data?.source || 'TESTNET'
                        };

                        // Only update if matches our account
                        if (!data.exchange_account_id || data.exchange_account_id === accountId) {
                            setPrices(prev => ({
                                ...prev,
                                [priceData.symbol]: priceData
                            }));
                            setLastUpdate(Date.now());
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
    }, [accountId, symbols]);

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
        if (autoConnect && accountId) {
            // Disconnect previous connection
            disconnect();
            // Clear prices for new account
            clearPrices();
            // Connect to new account
            connect();
        } else if (!accountId) {
            disconnect();
            clearPrices();
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
