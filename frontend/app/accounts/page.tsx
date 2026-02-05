'use client';

/**
 * Exchange Accounts Page - CRYPTOBOSS 2.0
 * 
 * Features:
 * - Account list with reset/delete buttons
 * - Delete means archive (keep for audit)
 * - Reset means DELETE ALL TRADES for that account
 */

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';

export default function AccountsPage() {
    const { user, accounts, activeAccount, selectAccount, createAccount, isLoading, isAuthenticated, refreshAccounts } = useAuth();
    const [showAddForm, setShowAddForm] = useState(false);
    const [formData, setFormData] = useState({
        exchange_name: 'binance',
        environment: 'TESTNET',
        api_key: '',
        api_secret: '',
        label: ''
    });
    const [error, setError] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [resetModal, setResetModal] = useState<{ show: boolean; accountId: string; accountLabel: string }>({ show: false, accountId: '', accountLabel: '' });
    const [resetReason, setResetReason] = useState('');
    const [isResetting, setIsResetting] = useState(false);
    const [resetResult, setResetResult] = useState<{ success: boolean; message: string } | null>(null);
    const router = useRouter();

    useEffect(() => {
        if (!isLoading && !isAuthenticated) {
            router.push('/auth/login');
        }
    }, [isLoading, isAuthenticated, router]);

    const handleCreateAccount = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsSubmitting(true);

        const result = await createAccount(formData);

        if (result.success) {
            setShowAddForm(false);
            setFormData({
                exchange_name: 'binance',
                environment: 'TESTNET',
                api_key: '',
                api_secret: '',
                label: ''
            });
        } else {
            setError(result.error || 'Failed to create account');
        }

        setIsSubmitting(false);
    };

    const handleSelectAccount = async (accountId: string) => {
        await selectAccount(accountId);
        router.push('/');
    };

    const handleResetAccount = async () => {
        if (resetReason.length < 10) {
            setError('Reason must be at least 10 characters');
            return;
        }

        setIsResetting(true);
        setError('');

        try {
            const token = localStorage.getItem('auth_token');
            const response = await fetch(`http://localhost:8000/api/accounts/${resetModal.accountId}/reset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    confirm: true,
                    reason: resetReason
                })
            });

            const data = await response.json();

            if (response.ok) {
                setResetResult({
                    success: true,
                    message: data.data?.message || `Account reset. ${data.data?.trades_deleted || 0} trades deleted.`
                });
                // Refresh accounts after reset
                if (refreshAccounts) {
                    await refreshAccounts();
                }
            } else {
                setResetResult({
                    success: false,
                    message: data.detail || 'Failed to reset account'
                });
            }
        } catch (e: any) {
            setResetResult({
                success: false,
                message: e.message || 'Network error'
            });
        }

        setIsResetting(false);
    };

    const closeResetModal = () => {
        setResetModal({ show: false, accountId: '', accountLabel: '' });
        setResetReason('');
        setResetResult(null);
        setError('');
    };

    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-900">
                <div className="text-white">Loading...</div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 p-8">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Exchange Accounts</h1>
                        <p className="text-gray-400 mt-1">
                            Logged in as {user?.email}
                        </p>
                    </div>
                    <button
                        onClick={() => setShowAddForm(true)}
                        className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition flex items-center gap-2"
                    >
                        <span className="text-xl">+</span> Add Account
                    </button>
                </div>

                {/* Active Account Banner */}
                {activeAccount && (
                    <div className="mb-6 p-4 bg-green-500/20 border border-green-500/50 rounded-xl">
                        <div className="flex items-center gap-3">
                            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                            <div>
                                <p className="text-green-300 font-medium">Active Account</p>
                                <p className="text-white">{activeAccount.label}</p>
                            </div>
                            <span className={`ml-auto px-3 py-1 rounded-full text-sm font-medium ${activeAccount.environment === 'LIVE'
                                ? 'bg-red-500/20 text-red-300 border border-red-500/50'
                                : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/50'
                                }`}>
                                {activeAccount.environment}
                            </span>
                        </div>
                    </div>
                )}

                {/* No Accounts Message */}
                {accounts.length === 0 && !showAddForm && (
                    <div className="text-center py-16">
                        <div className="text-6xl mb-4">🔑</div>
                        <h2 className="text-xl text-white mb-2">No Exchange Accounts</h2>
                        <p className="text-gray-400 mb-6">Add your first exchange account to start trading</p>
                        <button
                            onClick={() => setShowAddForm(true)}
                            className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition"
                        >
                            Add Exchange Account
                        </button>
                    </div>
                )}

                {/* Account List */}
                {accounts.length > 0 && (
                    <div className="grid gap-4">
                        {accounts.map((account) => (
                            <div
                                key={account.exchange_account_id}
                                className={`p-6 bg-gray-800/50 border rounded-xl transition ${activeAccount?.exchange_account_id === account.exchange_account_id
                                    ? 'border-blue-500'
                                    : 'border-gray-700'
                                    }`}
                            >
                                <div
                                    className="flex items-center justify-between cursor-pointer"
                                    onClick={() => handleSelectAccount(account.exchange_account_id)}
                                >
                                    <div>
                                        <h3 className="text-lg font-medium text-white">{account.label}</h3>
                                        <p className="text-gray-400 text-sm">
                                            {account.exchange_name.toUpperCase()} • Created {new Date(account.created_at).toLocaleDateString()}
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${account.environment === 'LIVE'
                                            ? 'bg-red-500/20 text-red-300'
                                            : 'bg-yellow-500/20 text-yellow-300'
                                            }`}>
                                            {account.environment}
                                        </span>
                                        {activeAccount?.exchange_account_id === account.exchange_account_id && (
                                            <span className="text-green-400">✓ Active</span>
                                        )}
                                    </div>
                                </div>

                                {/* Action Buttons */}
                                <div className="mt-4 pt-4 border-t border-gray-700/50 flex gap-3">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setResetModal({
                                                show: true,
                                                accountId: account.exchange_account_id,
                                                accountLabel: account.label
                                            });
                                        }}
                                        className="px-4 py-2 bg-yellow-500/20 text-yellow-300 border border-yellow-500/50 rounded-lg text-sm hover:bg-yellow-500/30 transition"
                                    >
                                        🗑️ Reset Data
                                    </button>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            if (confirm('Are you sure you want to archive this account? This cannot be undone.')) {
                                                // TODO: Implement delete
                                                alert('Delete functionality coming soon');
                                            }
                                        }}
                                        className="px-4 py-2 bg-red-500/20 text-red-300 border border-red-500/50 rounded-lg text-sm hover:bg-red-500/30 transition"
                                    >
                                        📦 Archive Account
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Add Account Form */}
                {showAddForm && (
                    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
                        <div className="bg-gray-800 border border-gray-700 rounded-2xl p-8 max-w-md w-full mx-4">
                            <h2 className="text-2xl font-semibold text-white mb-6">Add Exchange Account</h2>

                            {error && (
                                <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm">
                                    {error}
                                </div>
                            )}

                            <form onSubmit={handleCreateAccount} className="space-y-5">
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">
                                        Account Label
                                    </label>
                                    <input
                                        type="text"
                                        value={formData.label}
                                        onChange={(e) => setFormData({ ...formData, label: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                                        placeholder="Main Trading Account"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">
                                        Environment
                                    </label>
                                    <select
                                        value={formData.environment}
                                        onChange={(e) => setFormData({ ...formData, environment: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                                    >
                                        <option value="TESTNET">TESTNET (Recommended for testing)</option>
                                        <option value="LIVE">LIVE (Real money)</option>
                                    </select>
                                    {formData.environment === 'LIVE' && (
                                        <p className="mt-2 text-red-400 text-sm">
                                            ⚠️ LIVE mode uses real money. Proceed with caution.
                                        </p>
                                    )}
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">
                                        API Key
                                    </label>
                                    <input
                                        type="text"
                                        value={formData.api_key}
                                        onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition font-mono"
                                        placeholder="Enter your API key"
                                        required
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">
                                        API Secret
                                    </label>
                                    <input
                                        type="password"
                                        value={formData.api_secret}
                                        onChange={(e) => setFormData({ ...formData, api_secret: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition font-mono"
                                        placeholder="Enter your API secret"
                                        required
                                    />
                                </div>

                                <div className="flex gap-3 pt-4">
                                    <button
                                        type="button"
                                        onClick={() => setShowAddForm(false)}
                                        className="flex-1 py-3 px-4 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="submit"
                                        disabled={isSubmitting}
                                        className="flex-1 py-3 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
                                    >
                                        {isSubmitting ? 'Creating...' : 'Create Account'}
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {/* Reset Account Modal */}
                {resetModal.show && (
                    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
                        <div className="bg-gray-800 border border-gray-700 rounded-2xl p-8 max-w-md w-full mx-4">
                            <h2 className="text-2xl font-semibold text-white mb-2">Reset Account Data</h2>
                            <p className="text-gray-400 mb-6">
                                This will DELETE all trades and analytics for: <strong className="text-white">{resetModal.accountLabel}</strong>
                            </p>

                            {error && (
                                <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm">
                                    {error}
                                </div>
                            )}

                            {resetResult && (
                                <div className={`mb-4 p-3 ${resetResult.success ? 'bg-green-500/20 border-green-500/50 text-green-300' : 'bg-red-500/20 border-red-500/50 text-red-300'} border rounded-lg text-sm`}>
                                    {resetResult.message}
                                </div>
                            )}

                            {!resetResult && (
                                <>
                                    <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                                        <h3 className="text-yellow-300 font-medium mb-2">⚠️ What will be deleted:</h3>
                                        <ul className="text-yellow-200/80 text-sm space-y-1">
                                            <li>• All trade history</li>
                                            <li>• PnL history</li>
                                            <li>• Analytics data</li>
                                        </ul>
                                        <h3 className="text-green-300 font-medium mt-4 mb-2">✅ What stays:</h3>
                                        <ul className="text-green-200/80 text-sm space-y-1">
                                            <li>• Your account</li>
                                            <li>• API keys</li>
                                            <li>• Other exchange accounts</li>
                                        </ul>
                                    </div>

                                    <div className="mb-6">
                                        <label className="block text-sm font-medium text-gray-300 mb-2">
                                            Reason for reset (required, min 10 chars)
                                        </label>
                                        <textarea
                                            value={resetReason}
                                            onChange={(e) => setResetReason(e.target.value)}
                                            className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-yellow-500 transition resize-none"
                                            placeholder="e.g., Starting fresh for new strategy"
                                            rows={3}
                                        />
                                    </div>

                                    <div className="flex gap-3">
                                        <button
                                            type="button"
                                            onClick={closeResetModal}
                                            className="flex-1 py-3 px-4 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
                                        >
                                            Cancel
                                        </button>
                                        <button
                                            type="button"
                                            onClick={handleResetAccount}
                                            disabled={isResetting || resetReason.length < 10}
                                            className="flex-1 py-3 px-4 bg-yellow-500 text-black rounded-lg hover:bg-yellow-400 transition disabled:opacity-50 font-medium"
                                        >
                                            {isResetting ? 'Resetting...' : 'Confirm Reset'}
                                        </button>
                                    </div>
                                </>
                            )}

                            {resetResult && (
                                <button
                                    type="button"
                                    onClick={closeResetModal}
                                    className="w-full py-3 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
                                >
                                    Close
                                </button>
                            )}
                        </div>
                    </div>
                )}

                {/* Back Link */}
                <div className="mt-8 text-center">
                    <Link href="/" className="text-gray-400 hover:text-white transition">
                        ← Back to Dashboard
                    </Link>
                </div>
            </div>
        </div>
    );
}
