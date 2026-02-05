'use client';

/**
 * Positions Page - CRYPTOBOSS vFINAL
 * 
 * Purpose: Trade transparency - shows ONLY backend data
 * Rules:
 * - NO mock data - fetch from /api/positions
 * - Empty state for new users
 * - Re-fetch on account change
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Position {
    id: number;
    symbol: string;
    side: string;
    entryPrice: number;
    currentPrice: number;
    size: number;
    exposure: number;
    unrealizedPnL: number;
    pnlPercent: number;
    entryTime: string;
    entryReason?: string;
    stopLoss?: number;
    takeProfit?: number;
}

interface ClosedPosition {
    id: number;
    symbol: string;
    side: string;
    entryPrice: number;
    exitPrice: number;
    size: number;
    realizedPnL: number;
    pnlPercent: number;
    entryTime: string;
    exitTime: string;
    exitReason?: string;
}

function PositionCard({ position }: { position: Position }) {
    const pnlColor = position.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400';
    const sideColor = position.side === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400';

    return (
        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <span className="text-white font-medium">{position.symbol}</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${sideColor}`}>{position.side}</span>
                </div>
                <span className={`font-medium ${pnlColor}`}>
                    {position.unrealizedPnL >= 0 ? '+' : ''}${position.unrealizedPnL.toFixed(2)}
                    <span className="text-sm ml-1">({position.pnlPercent.toFixed(2)}%)</span>
                </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 text-sm">
                <div>
                    <span className="text-gray-500 block">Entry Price</span>
                    <span className="text-white">${position.entryPrice.toLocaleString()}</span>
                </div>
                <div>
                    <span className="text-gray-500 block">Current Price</span>
                    <span className="text-white">${position.currentPrice.toLocaleString()}</span>
                </div>
                <div>
                    <span className="text-gray-500 block">Size</span>
                    <span className="text-white">{position.size}</span>
                </div>
                <div>
                    <span className="text-gray-500 block">Exposure</span>
                    <span className="text-white">${position.exposure.toLocaleString()}</span>
                </div>
            </div>

            {position.entryReason && (
                <div className="bg-gray-800/50 rounded-md p-3 mb-4">
                    <span className="text-gray-500 block text-xs mb-1">Entry Reasoning</span>
                    <p className="text-sm text-gray-400">{position.entryReason}</p>
                </div>
            )}

            {(position.stopLoss || position.takeProfit) && (
                <div className="grid grid-cols-2 gap-4 text-sm">
                    {position.stopLoss && (
                        <div>
                            <span className="text-gray-500 block">Stop Loss</span>
                            <span className="text-red-400">${position.stopLoss.toLocaleString()}</span>
                        </div>
                    )}
                    {position.takeProfit && (
                        <div>
                            <span className="text-gray-500 block">Take Profit</span>
                            <span className="text-green-400">${position.takeProfit.toLocaleString()}</span>
                        </div>
                    )}
                </div>
            )}

            <div className="mt-4 pt-4 border-t border-gray-700">
                <span className="text-xs text-gray-600">Opened: {position.entryTime}</span>
            </div>
        </div>
    );
}

export default function PositionsPage() {
    const { activeAccount, token } = useAuth();
    const [positions, setPositions] = useState<Position[]>([]);
    const [closedPositions, setClosedPositions] = useState<ClosedPosition[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchPositions = useCallback(async () => {
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/positions`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch positions');
            }

            const data = await response.json();

            // Backend returns empty array = new account with no positions
            setPositions(data.data?.positions || []);
            setClosedPositions(data.data?.closed_today || []);
            setError(null);
        } catch (e: any) {
            console.error('Positions fetch error:', e);
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }, [token]);

    // Fetch on mount and when account changes
    useEffect(() => {
        setPositions([]);
        setClosedPositions([]);
        setLoading(true);
        fetchPositions();
    }, [activeAccount, fetchPositions]);

    // Refresh every 10 seconds
    useEffect(() => {
        const interval = setInterval(fetchPositions, 10000);
        return () => clearInterval(interval);
    }, [fetchPositions]);

    const totalExposure = positions.reduce((sum, p) => sum + (p.exposure || 0), 0);
    const totalUnrealizedPnL = positions.reduce((sum, p) => sum + (p.unrealizedPnL || 0), 0);

    return (
        <div className="p-6 space-y-6">
            {/* Page Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-1">Positions</h1>
                <p className="text-gray-400 text-sm">
                    {activeAccount
                        ? `Account: ${activeAccount.label}`
                        : 'No account selected'}
                </p>
            </div>

            {/* Loading State */}
            {loading && (
                <div className="text-center py-12">
                    <div className="text-gray-400">Loading positions...</div>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-400">
                    Error: {error}
                </div>
            )}

            {/* No Account Selected */}
            {!loading && !activeAccount && (
                <div className="text-center py-12">
                    <div className="text-5xl mb-4">🔐</div>
                    <div className="text-xl text-white mb-2">No Account Selected</div>
                    <div className="text-gray-400">
                        Please select an exchange account to view positions
                    </div>
                </div>
            )}

            {/* Has Account - Show Data */}
            {!loading && activeAccount && (
                <>
                    {/* Summary Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <span className="text-gray-400 text-sm">Open Positions</span>
                            <span className="text-2xl font-bold text-white block mt-1">{positions.length}</span>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <span className="text-gray-400 text-sm">Total Exposure</span>
                            <span className="text-2xl font-bold text-white block mt-1">
                                ${totalExposure.toLocaleString()}
                            </span>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <span className="text-gray-400 text-sm">Unrealized P&L</span>
                            <span className={`text-2xl font-bold block mt-1 ${totalUnrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'
                                }`}>
                                {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
                            </span>
                        </div>
                        <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl p-4 text-center">
                            <span className="text-gray-400 text-sm">Closed Today</span>
                            <span className="text-2xl font-bold text-white block mt-1">{closedPositions.length}</span>
                        </div>
                    </div>

                    {/* Open Positions */}
                    <div>
                        <h2 className="text-xl font-semibold text-white mb-4">Open Positions</h2>
                        {positions.length > 0 ? (
                            <div className="space-y-4">
                                {positions.map((position) => (
                                    <PositionCard key={position.id} position={position} />
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-12 bg-[#1d2229] border border-[#2d3640] rounded-xl">
                                <div className="text-5xl mb-4">📭</div>
                                <div className="text-xl text-white mb-2">No Open Positions</div>
                                <div className="text-gray-400">
                                    The system has no active positions. This is a normal state.
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Closed Positions Today */}
                    <div className="mt-8">
                        <h2 className="text-xl font-semibold text-white mb-4">Closed Today</h2>
                        {closedPositions.length > 0 ? (
                            <div className="bg-[#1d2229] border border-[#2d3640] rounded-xl overflow-hidden">
                                <table className="w-full">
                                    <thead className="bg-gray-800/50">
                                        <tr className="text-left text-gray-400 text-sm">
                                            <th className="p-4">Symbol</th>
                                            <th className="p-4">Side</th>
                                            <th className="p-4">Entry</th>
                                            <th className="p-4">Exit</th>
                                            <th className="p-4">P&L</th>
                                            <th className="p-4">Reason</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {closedPositions.map((pos) => (
                                            <tr key={pos.id} className="border-t border-gray-700">
                                                <td className="p-4 text-white">{pos.symbol}</td>
                                                <td className="p-4">
                                                    <span className={`px-2 py-0.5 rounded text-xs ${pos.side === 'LONG'
                                                            ? 'bg-green-500/20 text-green-400'
                                                            : 'bg-red-500/20 text-red-400'
                                                        }`}>
                                                        {pos.side}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-white">${pos.entryPrice.toLocaleString()}</td>
                                                <td className="p-4 text-white">${pos.exitPrice.toLocaleString()}</td>
                                                <td className={`p-4 ${pos.realizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {pos.realizedPnL >= 0 ? '+' : ''}${pos.realizedPnL.toFixed(2)}
                                                </td>
                                                <td className="p-4 text-gray-400">{pos.exitReason || '-'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className="text-center py-8 bg-[#1d2229] border border-[#2d3640] rounded-xl">
                                <div className="text-gray-400">No positions closed today</div>
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
