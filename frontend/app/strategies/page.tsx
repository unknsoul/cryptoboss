'use client';

/**
 * Strategies Page - CRYPTOBOSS vFINAL
 * 
 * Purpose: Display strategy status from backend
 * Rules:
 * - NO mock data - fetch from /api/strategies
 * - Empty state for new accounts
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Strategy {
    id: string;
    name: string;
    enabled: boolean;
    healthScore: number;
    recentDecay: number;
    wins: number;
    losses: number;
    lastTrade?: string;
}

export default function StrategiesPage() {
    const { activeAccount, token } = useAuth();
    const [strategies, setStrategies] = useState<Strategy[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchStrategies = useCallback(async () => {
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/strategies`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (!response.ok) throw new Error('Failed to fetch strategies');
            const data = await response.json();
            setStrategies(data.data?.strategies || []);
            setError(null);
        } catch (e: any) {
            console.error('Strategies fetch error:', e);
            setError(e.message);
            setStrategies([]);
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        setStrategies([]);
        setLoading(true);
        fetchStrategies();
    }, [activeAccount, fetchStrategies]);

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-1">Strategies</h1>
                <p className="text-gray-400 text-sm">
                    {activeAccount ? `Account: ${activeAccount.label}` : 'No account selected'}
                </p>
            </div>

            {loading && <div className="text-center py-12 text-gray-400">Loading strategies...</div>}
            {error && <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-400">Error: {error}</div>}

            {!loading && !activeAccount && (
                <div className="text-center py-12">
                    <div className="text-5xl mb-4">🔐</div>
                    <div className="text-xl text-white mb-2">No Account Selected</div>
                    <div className="text-gray-400">Please select an exchange account</div>
                </div>
            )}

            {!loading && activeAccount && strategies.length === 0 && (
                <div className="text-center py-12 bg-[#1d2229] border border-[#2d3640] rounded-xl">
                    <div className="text-5xl mb-4">📋</div>
                    <div className="text-xl text-white mb-2">No Strategies Configured</div>
                    <div className="text-gray-400">This account has no active strategies</div>
                </div>
            )}

            {!loading && activeAccount && strategies.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {strategies.map((strategy) => (
                        <div key={strategy.id} className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <span className="text-lg font-medium text-white">{strategy.name}</span>
                                <span className={`px-2 py-0.5 rounded text-xs ${strategy.enabled ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'
                                    }`}>
                                    {strategy.enabled ? 'ENABLED' : 'DISABLED'}
                                </span>
                            </div>

                            <div className="space-y-3">
                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-gray-400">Health Score</span>
                                        <span className={`${strategy.healthScore >= 0.7 ? 'text-green-400' : strategy.healthScore >= 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {(strategy.healthScore * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full ${strategy.healthScore >= 0.7 ? 'bg-green-500' : strategy.healthScore >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                            style={{ width: `${strategy.healthScore * 100}%` }}
                                        />
                                    </div>
                                </div>

                                <div className="flex justify-between text-sm">
                                    <span className="text-gray-400">Win/Loss</span>
                                    <span>
                                        <span className="text-green-400">{strategy.wins}W</span>
                                        {' / '}
                                        <span className="text-red-400">{strategy.losses}L</span>
                                    </span>
                                </div>

                                {strategy.lastTrade && (
                                    <div className="text-xs text-gray-600">
                                        Last trade: {new Date(strategy.lastTrade).toLocaleString()}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
                <div className="flex items-start gap-3 text-sm text-gray-400">
                    <span className="text-blue-400">ℹ️</span>
                    <p>Strategies propose, they don't execute. All trades go through the execution flow.</p>
                </div>
            </div>
        </div>
    );
}
