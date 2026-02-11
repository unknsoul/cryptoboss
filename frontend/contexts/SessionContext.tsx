'use client';

/**
 * Session Context - Central session lifecycle management for frontend
 * 
 * CRYPTOBOSS 2.0: PAPER TRADING REMOVED
 * Only TESTNET and LIVE environments are supported.
 * 
 * Provides:
 * - Current session ID
 * - Trading environment (testnet/live)
 * - Connection status
 * - Environment switching with confirmation
 * - Session reset functionality
 */

import React, { createContext, useContext, useReducer, useEffect, useCallback, ReactNode } from 'react';

// Types - PAPER REMOVED
export type TradingMode = 'testnet' | 'live';
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ApiConfig {
    apiKey: string;
    apiSecret: string;
    isValidated: boolean;
    validatedAt?: string;
}

export interface SessionState {
    sessionId: string | null;
    mode: TradingMode;
    connectionStatus: ConnectionStatus;
    apiConfig: ApiConfig | null;
    isInitialized: boolean;
    lastError: string | null;
    balances: Record<string, number>;
    createdAt: string | null;
    activeAccount: { exchange_account_id: string; label: string; environment: string } | null;
}

type SessionAction =
    | { type: 'INIT_SESSION'; payload: { sessionId: string; mode: TradingMode; createdAt: string } }
    | { type: 'SET_MODE'; payload: TradingMode }
    | { type: 'SET_CONNECTION_STATUS'; payload: ConnectionStatus }
    | { type: 'SET_API_CONFIG'; payload: ApiConfig | null }
    | { type: 'SET_BALANCES'; payload: Record<string, number> }
    | { type: 'SET_ACTIVE_ACCOUNT'; payload: { exchange_account_id: string; label: string; environment: string } | null }
    | { type: 'SET_ERROR'; payload: string | null }
    | { type: 'RESET_SESSION' }
    | { type: 'CLEAR_ALL' };

const initialState: SessionState = {
    sessionId: null,
    mode: 'testnet',  // PAPER REMOVED - default to testnet
    connectionStatus: 'disconnected',
    apiConfig: null,
    isInitialized: false,
    lastError: null,
    balances: {},
    createdAt: null,
    activeAccount: null,
};

function sessionReducer(state: SessionState, action: SessionAction): SessionState {
    switch (action.type) {
        case 'INIT_SESSION':
            return {
                ...initialState,
                sessionId: action.payload.sessionId,
                mode: action.payload.mode,
                createdAt: action.payload.createdAt,
                isInitialized: true,
                connectionStatus: 'disconnected',  // Always need to connect to exchange
            };
        case 'SET_MODE':
            return { ...state, mode: action.payload };
        case 'SET_CONNECTION_STATUS':
            return { ...state, connectionStatus: action.payload };
        case 'SET_API_CONFIG':
            return { ...state, apiConfig: action.payload };
        case 'SET_BALANCES':
            return { ...state, balances: action.payload };
        case 'SET_ACTIVE_ACCOUNT':
            // CRYPTOBOSS 2.0: Clear balances when switching accounts
            return { ...state, activeAccount: action.payload, balances: {} };
        case 'SET_ERROR':
            return { ...state, lastError: action.payload };
        case 'RESET_SESSION':
            return {
                ...initialState,
                sessionId: generateSessionId(),
                mode: state.mode,
                createdAt: new Date().toISOString(),
                isInitialized: true,
            };
        case 'CLEAR_ALL':
            return initialState;
        default:
            return state;
    }
}

// Generate a new session ID
function generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Context
interface SessionContextType {
    state: SessionState;
    switchMode: (newMode: TradingMode, apiConfig?: ApiConfig) => Promise<boolean>;
    resetSession: () => void;
    validateApiKeys: (apiKey: string, apiSecret: string) => Promise<{ success: boolean; message: string; balances?: Record<string, number> }>;
    clearError: () => void;
    setBalances: (balances: Record<string, number>) => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

// API base URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Provider
export function SessionProvider({ children }: { children: ReactNode }) {
    const [state, dispatch] = useReducer(sessionReducer, initialState);

    // Initialize session on mount - default to TESTNET
    useEffect(() => {
        const sessionId = generateSessionId();
        dispatch({
            type: 'INIT_SESSION',
            payload: {
                sessionId,
                mode: 'testnet',  // PAPER REMOVED
                createdAt: new Date().toISOString(),
            },
        });
    }, []);;

    // Switch trading environment
    const switchMode = useCallback(async (newMode: TradingMode, apiConfig?: ApiConfig): Promise<boolean> => {
        try {
            dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'connecting' });
            dispatch({ type: 'SET_ERROR', payload: null });

            // PAPER MODE REMOVED - Always require API credentials
            if (!apiConfig || !apiConfig.apiKey || !apiConfig.apiSecret) {
                dispatch({ type: 'SET_ERROR', payload: 'API credentials required' });
                dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'error' });
                return false;
            }

            // Call backend to validate and switch mode
            const response = await fetch(`${API_BASE}/api/session/switch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: newMode,
                    api_key: apiConfig.apiKey,
                    api_secret: apiConfig.apiSecret,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.message || 'Failed to switch mode');
            }

            const data = await response.json();

            // Initialize new session with backend session ID
            dispatch({
                type: 'INIT_SESSION',
                payload: {
                    sessionId: data.session_id,
                    mode: newMode,
                    createdAt: data.created_at,
                },
            });

            dispatch({
                type: 'SET_API_CONFIG',
                payload: { ...apiConfig, isValidated: true, validatedAt: new Date().toISOString() }
            });

            if (data.balances) {
                dispatch({ type: 'SET_BALANCES', payload: data.balances });
            }

            dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'connected' });
            return true;

        } catch (error) {
            const message = error instanceof Error ? error.message : 'Unknown error';
            dispatch({ type: 'SET_ERROR', payload: message });
            dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'error' });
            return false;
        }
    }, []);

    // Reset session (clear all data, generate new session ID)
    const resetSession = useCallback(() => {
        dispatch({ type: 'RESET_SESSION' });
    }, []);

    // Validate API keys without switching mode
    const validateApiKeys = useCallback(async (
        apiKey: string,
        apiSecret: string
    ): Promise<{ success: boolean; message: string; balances?: Record<string, number> }> => {
        try {
            const response = await fetch(`${API_BASE}/api/validate-keys`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    api_key: apiKey,
                    api_secret: apiSecret,
                    testnet: state.mode === 'testnet',
                }),
            });

            const data = await response.json();
            return {
                success: data.success,
                message: data.message,
                balances: data.balances,
            };
        } catch (error) {
            return {
                success: false,
                message: error instanceof Error ? error.message : 'Validation failed',
            };
        }
    }, [state.mode]);

    // Clear error
    const clearError = useCallback(() => {
        dispatch({ type: 'SET_ERROR', payload: null });
    }, []);

    // Set balances
    const setBalances = useCallback((balances: Record<string, number>) => {
        dispatch({ type: 'SET_BALANCES', payload: balances });
    }, []);

    const value: SessionContextType = {
        state,
        switchMode,
        resetSession,
        validateApiKeys,
        clearError,
        setBalances,
    };

    return (
        <SessionContext.Provider value={value}>
            {children}
        </SessionContext.Provider>
    );
}

// Hook
export function useSession(): SessionContextType {
    const context = useContext(SessionContext);
    if (context === undefined) {
        throw new Error('useSession must be used within a SessionProvider');
    }
    return context;
}
