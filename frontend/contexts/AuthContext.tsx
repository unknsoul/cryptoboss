'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
    user_id: string;
    email: string;
    created_at: string;
    is_active: boolean;
}

interface ExchangeAccount {
    exchange_account_id: string;
    user_id: string;
    exchange_name: string;
    environment: string;
    label: string;
    created_at: string;
    last_validated_at: string | null;
    is_active: boolean;
    api_key_fingerprint?: string;
}

interface AuthContextType {
    user: User | null;
    token: string | null;
    activeAccount: ExchangeAccount | null;
    accounts: ExchangeAccount[];
    isLoading: boolean;
    isAuthenticated: boolean;
    login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
    signup: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
    logout: () => void;
    selectAccount: (accountId: string) => Promise<void>;
    refreshAccounts: () => Promise<void>;
    createAccount: (data: CreateAccountData) => Promise<{ success: boolean; error?: string }>;
}

interface CreateAccountData {
    exchange_name: string;
    environment: string;
    api_key: string;
    api_secret: string;
    label?: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [activeAccount, setActiveAccount] = useState<ExchangeAccount | null>(null);
    const [accounts, setAccounts] = useState<ExchangeAccount[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Load token from localStorage on mount
    useEffect(() => {
        const storedToken = localStorage.getItem('cryptoboss_token');
        if (storedToken) {
            setToken(storedToken);
            fetchUser(storedToken);
        } else {
            setIsLoading(false);
        }
    }, []);

    const fetchUser = async (authToken: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/auth/me`, {
                headers: { Authorization: `Bearer ${authToken}` }
            });
            if (res.ok) {
                const data = await res.json();
                setUser(data.data.user);
                await refreshAccounts(authToken);
                await fetchActiveAccount(authToken);
            } else {
                // Token invalid
                localStorage.removeItem('cryptoboss_token');
                setToken(null);
            }
        } catch (error) {
            console.error('Failed to fetch user:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const fetchActiveAccount = async (authToken: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/accounts/active`, {
                headers: { Authorization: `Bearer ${authToken}` }
            });
            if (res.ok) {
                const data = await res.json();
                if (data.data.active) {
                    setActiveAccount(data.data.account);
                }
            }
        } catch (error) {
            console.error('Failed to fetch active account:', error);
        }
    };

    const refreshAccounts = async (authToken?: string) => {
        const tokenToUse = authToken || token;
        if (!tokenToUse) return;

        try {
            const res = await fetch(`${API_BASE}/api/accounts/list`, {
                headers: { Authorization: `Bearer ${tokenToUse}` }
            });
            if (res.ok) {
                const data = await res.json();
                setAccounts(data.data.accounts || []);
            }
        } catch (error) {
            console.error('Failed to fetch accounts:', error);
        }
    };

    const login = async (email: string, password: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();

            if (res.ok && data.data.success) {
                setToken(data.data.token);
                setUser(data.data.user);
                localStorage.setItem('cryptoboss_token', data.data.token);
                await refreshAccounts(data.data.token);
                return { success: true };
            } else {
                return { success: false, error: data.detail || 'Login failed' };
            }
        } catch (error) {
            return { success: false, error: 'Network error' };
        }
    };

    const signup = async (email: string, password: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/auth/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();

            if (res.ok && data.data.success) {
                setToken(data.data.token);
                setUser(data.data.user);
                localStorage.setItem('cryptoboss_token', data.data.token);
                return { success: true };
            } else {
                return { success: false, error: data.detail || 'Signup failed' };
            }
        } catch (error) {
            return { success: false, error: 'Network error' };
        }
    };

    const logout = () => {
        setUser(null);
        setToken(null);
        setActiveAccount(null);
        setAccounts([]);
        localStorage.removeItem('cryptoboss_token');
    };

    const selectAccount = async (accountId: string) => {
        if (!token) return;

        try {
            const res = await fetch(`${API_BASE}/api/accounts/select`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify({ exchange_account_id: accountId })
            });

            if (res.ok) {
                const data = await res.json();

                // v1.0.1: CRITICAL - Full state reset on account switch
                console.log('🔄 Account switched - triggering full reset');

                // 1. Clear session storage (cached responses)
                sessionStorage.clear();

                // 2. Dispatch custom event for other components to reset
                window.dispatchEvent(new CustomEvent('accountSwitched', {
                    detail: {
                        accountId: accountId,
                        isNewAccount: data.data.is_new_account,
                        environment: data.data.account.environment,
                        actions: ['CLEAR_ALL_STORES', 'RESET_CHARTS', 'DROP_CACHED_RESPONSES']
                    }
                }));

                // 3. Update active account state
                setActiveAccount(data.data.account);

                // 4. Store active account ID in localStorage for persistence
                localStorage.setItem('cryptoboss_active_account', accountId);

                console.log(`✅ Switched to account: ${accountId.substring(0, 8)}... (${data.data.is_new_account ? 'NEW' : 'existing'})`);
            }
        } catch (error) {
            console.error('Failed to select account:', error);
        }
    };

    const createAccount = async (data: CreateAccountData) => {
        if (!token) return { success: false, error: 'Not authenticated' };

        try {
            const res = await fetch(`${API_BASE}/api/accounts/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify(data)
            });

            const resData = await res.json();

            if (res.ok && resData.data.success) {
                await refreshAccounts();
                return { success: true };
            } else {
                return { success: false, error: resData.detail || 'Failed to create account' };
            }
        } catch (error) {
            return { success: false, error: 'Network error' };
        }
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                token,
                activeAccount,
                accounts,
                isLoading,
                isAuthenticated: !!user,
                login,
                signup,
                logout,
                selectAccount,
                refreshAccounts: () => refreshAccounts(),
                createAccount
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}
