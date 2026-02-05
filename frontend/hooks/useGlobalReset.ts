'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useAuth } from '@/contexts/AuthContext';

/**
 * Global Reset Event
 * 
 * Dispatched when account switches.
 * All components MUST listen and reset their state.
 */
export interface GlobalResetEvent {
    accountId: string;
    environment: string;
    isNewAccount: boolean;
    timestamp: number;
}

/**
 * Global reset hook for components
 * 
 * MANDATORY ACTIONS on account switch:
 * 1. Clear all UI stores
 * 2. Clear charts
 * 3. Clear tables
 * 4. Unsubscribe all websockets
 * 5. Block UI until fresh data arrives
 */
export function useGlobalReset(onReset?: () => void) {
    const { activeAccount } = useAuth();
    const [isResetting, setIsResetting] = useState(false);
    const [hasFreshData, setHasFreshData] = useState(false);
    const previousAccountRef = useRef<string | null>(null);

    // Handle account changes
    useEffect(() => {
        const currentAccountId = activeAccount?.exchange_account_id || null;

        if (previousAccountRef.current !== null &&
            previousAccountRef.current !== currentAccountId) {
            // Account changed - trigger reset
            console.log('🔄 Global reset triggered - account changed');
            setIsResetting(true);
            setHasFreshData(false);

            // Call reset callback
            if (onReset) {
                onReset();
            }

            // Clear after short delay
            setTimeout(() => {
                setIsResetting(false);
            }, 100);
        }

        previousAccountRef.current = currentAccountId;
    }, [activeAccount?.exchange_account_id, onReset]);

    // Listen for custom reset event
    useEffect(() => {
        const handleReset = (event: CustomEvent<GlobalResetEvent>) => {
            console.log('🔄 Global reset event received:', event.detail);
            setIsResetting(true);
            setHasFreshData(false);

            if (onReset) {
                onReset();
            }

            setTimeout(() => {
                setIsResetting(false);
            }, 100);
        };

        window.addEventListener('accountSwitched', handleReset as EventListener);
        return () => window.removeEventListener('accountSwitched', handleReset as EventListener);
    }, [onReset]);

    // Mark as having fresh data
    const markFreshData = useCallback(() => {
        setHasFreshData(true);
    }, []);

    return {
        isResetting,
        hasFreshData,
        markFreshData,
        shouldBlockUI: isResetting || (!hasFreshData && activeAccount !== null)
    };
}

/**
 * Hook for WebSocket management with auto-reconnect on account switch
 */
export function useAccountScopedWebSocket(url: string) {
    const { activeAccount } = useAuth();
    const wsRef = useRef<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<any>(null);

    // Close and reconnect on account switch
    useEffect(() => {
        if (!activeAccount?.exchange_account_id) {
            // No account - close connection
            if (wsRef.current) {
                console.log('📡 Closing WebSocket - no active account');
                wsRef.current.close();
                wsRef.current = null;
                setIsConnected(false);
            }
            return;
        }

        // Close existing connection
        if (wsRef.current) {
            console.log('📡 Closing WebSocket for account switch');
            wsRef.current.close();
        }

        // Open new connection
        console.log(`📡 Opening WebSocket for account ${activeAccount.exchange_account_id.substring(0, 8)}...`);
        const ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('📡 WebSocket connected');
            setIsConnected(true);

            // Send account context
            ws.send(JSON.stringify({
                type: 'subscribe',
                exchange_account_id: activeAccount.exchange_account_id,
                environment: activeAccount.environment
            }));
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Validate account match
                if (data.exchange_account_id &&
                    data.exchange_account_id !== activeAccount.exchange_account_id) {
                    console.warn('⚠️ Dropping WebSocket message - account mismatch');
                    return;
                }

                setLastMessage(data);
            } catch (e) {
                console.error('WebSocket message parse error:', e);
            }
        };

        ws.onclose = () => {
            console.log('📡 WebSocket disconnected');
            setIsConnected(false);
        };

        ws.onerror = (error) => {
            console.error('📡 WebSocket error:', error);
        };

        wsRef.current = ws;

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [activeAccount?.exchange_account_id, url]);

    const send = useCallback((data: any) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        }
    }, []);

    return {
        isConnected,
        lastMessage,
        send,
        accountId: activeAccount?.exchange_account_id
    };
}
