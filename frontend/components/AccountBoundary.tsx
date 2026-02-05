'use client';

import { useAuth } from '@/contexts/AuthContext';
import { useEffect, useState } from 'react';

/**
 * Account Boundary Component
 * 
 * Visual confirmation of which account is active.
 * Shows NEW ACCOUNT badge for accounts < 24 hours old.
 */
export function AccountBoundary() {
    const { activeAccount, isAuthenticated } = useAuth();
    const [isNew, setIsNew] = useState(false);

    useEffect(() => {
        if (activeAccount?.created_at) {
            const created = new Date(activeAccount.created_at);
            const now = new Date();
            const ageHours = (now.getTime() - created.getTime()) / (1000 * 60 * 60);
            setIsNew(ageHours < 24);
        }
    }, [activeAccount]);

    if (!isAuthenticated || !activeAccount) {
        return null;
    }

    return (
        <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg border border-gray-700 text-sm">
            {/* Environment Badge */}
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${activeAccount.environment === 'LIVE'
                    ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                    : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                }`}>
                {activeAccount.environment}
            </span>

            {/* Account ID (truncated) */}
            <span className="text-gray-400 font-mono text-xs">
                {activeAccount.exchange_account_id.substring(0, 8)}...
            </span>

            {/* NEW ACCOUNT Badge (first 24 hours) */}
            {isNew && (
                <span className="px-2 py-0.5 bg-blue-500/20 text-blue-300 border border-blue-500/30 rounded text-xs font-medium animate-pulse">
                    NEW ACCOUNT
                </span>
            )}

            {/* Account Label */}
            <span className="text-gray-300">
                {activeAccount.label}
            </span>
        </div>
    );
}


/**
 * Empty State Component
 * 
 * Shows when there's no data for the current account.
 * RULE: Tabs must not render without scoped data.
 */
interface EmptyStateProps {
    title?: string;
    message?: string;
    icon?: string;
}

export function EmptyAccountState({
    title = "No Data Available",
    message = "This is a new account. Data will appear once you start trading.",
    icon = "📊"
}: EmptyStateProps) {
    const { activeAccount } = useAuth();

    return (
        <div className="flex flex-col items-center justify-center p-12 text-center">
            <div className="text-5xl mb-4">{icon}</div>
            <h3 className="text-xl font-medium text-white mb-2">{title}</h3>
            <p className="text-gray-400 max-w-md mb-4">{message}</p>

            {activeAccount && (
                <div className="text-xs text-gray-500">
                    Account: {activeAccount.exchange_account_id.substring(0, 8)}... ({activeAccount.environment})
                </div>
            )}
        </div>
    );
}


/**
 * Data Scope Validator Hook
 * 
 * Validates that data matches the current active account.
 * HARD RULE: If mismatch, DROP DATA.
 */
export function useDataScope() {
    const { activeAccount } = useAuth();

    const validateScope = (data: any): boolean => {
        if (!activeAccount) return false;
        if (!data) return false;

        // Check if data has account ID and it matches
        const dataAccountId = data.exchange_account_id;
        if (dataAccountId && dataAccountId !== activeAccount.exchange_account_id) {
            console.warn(`⚠️ Data scope mismatch: expected ${activeAccount.exchange_account_id.substring(0, 8)}, got ${dataAccountId?.substring(0, 8)}`);
            return false;
        }

        return true;
    };

    const scopedFetch = async (url: string, options?: RequestInit): Promise<any> => {
        const response = await fetch(url, options);
        const data = await response.json();

        // Validate scope
        if (!validateScope(data)) {
            console.warn('Dropping data due to scope mismatch');
            return null;
        }

        return data;
    };

    return {
        activeAccountId: activeAccount?.exchange_account_id,
        validateScope,
        scopedFetch
    };
}
