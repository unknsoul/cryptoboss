'use client';

import { useMemo } from 'react';
import useSWR from 'swr';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { useAuth } from '@/contexts/AuthContext';
import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TradeCard {
    trade_id: string;
    symbol: string;
    direction: string;
    pnl_usdt: number;
    rr_achieved: number;
    duration_minutes: number;
    closed_at: string;
}

interface TodaySummary {
    date: string;
    trades: number;
    win_rate: number;
    profit_factor: number;
    avg_rr: number;
    total_pnl_usdt: number;
    best_trade: TradeCard | null;
    worst_trade: TradeCard | null;
    avg_hold_duration_minutes: number;
}

interface HourlySlot {
    hour: number;
    trades: number;
    win_rate: number;
    total_pnl_usdt: number;
    avg_pnl_usdt?: number;
}

interface HeatmapRow {
    day: string;
    hours: HourlySlot[];
}

interface SymbolPerformance {
    symbol: string;
    trades: number;
    win_rate: number;
    total_pnl_usdt: number;
    avg_pnl_usdt: number;
    avg_rr: number;
}

interface StrategyPerformance {
    strategy: string;
    trades: number;
    win_rate: number;
    profit_factor: number;
    total_pnl_usdt: number;
    avg_rr: number;
}

interface WeeklyPoint {
    week_start: string;
    weekly_pnl_usdt: number;
    cumulative_pnl_usdt: number;
    equity: number;
}

interface DrawdownPeriod {
    start: string;
    end: string;
    depth_pct: number;
    duration_trades: number;
}

interface AnalyticsPayload {
    today: TodaySummary;
    hourly: {
        hourly: HourlySlot[];
        heatmap: HeatmapRow[];
    };
    symbols: {
        symbols: SymbolPerformance[];
    };
    strategies: {
        strategies: StrategyPerformance[];
    };
    weekly: {
        points: WeeklyPoint[];
        initial_capital: number;
    };
    drawdowns: {
        periods: DrawdownPeriod[];
    };
}

async function analyticsFetcher([token]: [string]): Promise<AnalyticsPayload> {
    const headers = { Authorization: `Bearer ${token}` };
    const [todayRes, hourlyRes, symbolsRes, strategiesRes, weeklyRes, drawdownsRes] = await Promise.all([
        fetch(`${API_URL}/api/analytics/today`, { headers, cache: 'no-store' }),
        fetch(`${API_URL}/api/analytics/hourly-performance`, { headers, cache: 'no-store' }),
        fetch(`${API_URL}/api/analytics/symbol-performance`, { headers, cache: 'no-store' }),
        fetch(`${API_URL}/api/analytics/strategy-breakdown`, { headers, cache: 'no-store' }),
        fetch(`${API_URL}/api/analytics/weekly-equity`, { headers, cache: 'no-store' }),
        fetch(`${API_URL}/api/analytics/drawdown-periods`, { headers, cache: 'no-store' }),
    ]);

    const responses = [todayRes, hourlyRes, symbolsRes, strategiesRes, weeklyRes, drawdownsRes];
    const failed = responses.find((response) => !response.ok);
    if (failed) {
        throw new Error(`Analytics request failed with HTTP ${failed.status}`);
    }

    const [todayPayload, hourlyPayload, symbolsPayload, strategiesPayload, weeklyPayload, drawdownsPayload] = await Promise.all(
        responses.map((response) => response.json()),
    );

    return {
        today: unwrapApiData(todayPayload),
        hourly: unwrapApiData(hourlyPayload),
        symbols: unwrapApiData(symbolsPayload),
        strategies: unwrapApiData(strategiesPayload),
        weekly: unwrapApiData(weeklyPayload),
        drawdowns: unwrapApiData(drawdownsPayload),
    };
}

function StatCard({
    label,
    value,
    tone = 'text-[#e7e9ea]',
    helper,
}: {
    label: string;
    value: string;
    tone?: string;
    helper?: string;
}) {
    return (
        <div className="card">
            <div className="text-xs uppercase tracking-wider text-[#6b7280]">{label}</div>
            <div className={`mt-3 text-2xl font-semibold ${tone}`}>{value}</div>
            {helper && <div className="mt-2 text-xs text-[#8b98a5]">{helper}</div>}
        </div>
    );
}

function formatHour(hour: number): string {
    return `${hour.toString().padStart(2, '0')}:00`;
}

export default function AnalyticsPage() {
    const { token, activeAccount } = useAuth();

    const { data, error, isLoading } = useSWR(
        token ? [token] : null,
        analyticsFetcher,
        {
            refreshInterval: 10000,
            revalidateOnFocus: true,
            keepPreviousData: true,
        },
    );

    const bestTrade = data?.today.best_trade;
    const worstTrade = data?.today.worst_trade;
    const symbolChart = useMemo(
        () => (data?.symbols.symbols || []).map((item) => ({
            ...item,
            win_rate_pct: Number((item.win_rate * 100).toFixed(1)),
        })),
        [data],
    );

    if (!token) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Analytics</h1>
                    <p className="text-[#8b98a5] text-sm">Deep performance analytics for the active trading account</p>
                </div>
                <div className="card text-sm text-[#8b98a5]">Log in and select an account to view analytics.</div>
            </div>
        );
    }

    if (!activeAccount) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Analytics</h1>
                    <p className="text-[#8b98a5] text-sm">Deep performance analytics for the active trading account</p>
                </div>
                <div className="card text-sm text-[#8b98a5]">Select an exchange account to load performance analytics.</div>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Analytics</h1>
                    <p className="text-[#8b98a5] text-sm">Deep performance analytics for the active trading account</p>
                </div>
                <div className="card text-sm text-[#8b98a5]">Loading analytics...</div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="space-y-6">
                <div className="mb-8">
                    <h1 className="heading-lg mb-1">Analytics</h1>
                    <p className="text-[#8b98a5] text-sm">Deep performance analytics for the active trading account</p>
                </div>
                <div className="card text-sm text-[#d28383] border border-[#a65454]">
                    {error instanceof Error ? error.message : 'Failed to load analytics.'}
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="mb-8">
                <h1 className="heading-lg mb-1">Analytics</h1>
                <p className="text-[#8b98a5] text-sm">
                    Deep performance analytics for <span className="text-[#e7e9ea]">{activeAccount.label}</span>
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <StatCard
                    label="Today PnL"
                    value={`${data.today.total_pnl_usdt >= 0 ? '+' : ''}$${data.today.total_pnl_usdt.toFixed(2)}`}
                    tone={data.today.total_pnl_usdt >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]'}
                    helper={`${data.today.trades} trades today`}
                />
                <StatCard
                    label="Win Rate"
                    value={`${(data.today.win_rate * 100).toFixed(1)}%`}
                    helper={`Avg hold ${data.today.avg_hold_duration_minutes.toFixed(0)} min`}
                />
                <StatCard
                    label="Profit Factor"
                    value={data.today.profit_factor.toFixed(2)}
                    helper={`Avg R:R ${data.today.avg_rr.toFixed(2)}`}
                />
                <StatCard
                    label="Best Trade"
                    value={bestTrade ? `${bestTrade.symbol} ${bestTrade.direction}` : '--'}
                    helper={bestTrade ? `${bestTrade.rr_achieved.toFixed(2)}R in ${bestTrade.duration_minutes.toFixed(0)} min` : 'No closed trades today'}
                />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="card xl:col-span-2">
                    <div className="card-header">
                        <span className="card-title">Weekly Equity Curve</span>
                    </div>
                    <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data.weekly.points}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2d3640" />
                                <XAxis
                                    dataKey="week_start"
                                    stroke="#6b7280"
                                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-GB', { month: 'short', day: 'numeric' })}
                                />
                                <YAxis stroke="#6b7280" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#16181d', border: '1px solid #2d3640' }}
                                    formatter={(value: number) => [`$${Number(value).toFixed(2)}`, 'Equity']}
                                    labelFormatter={(value) => new Date(value).toLocaleDateString('en-GB')}
                                />
                                <Line type="monotone" dataKey="equity" stroke="#5b7a9d" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Drawdown Focus</span>
                    </div>
                    <div className="space-y-3 max-h-72 overflow-auto pr-1">
                        {worstTrade ? (
                            <div className="rounded-md border border-[#a65454] bg-[#a65454]/10 p-4">
                                <div className="text-sm text-[#8b98a5]">Worst Trade Today</div>
                                <div className="mt-2 text-[#e7e9ea] font-medium">{worstTrade.symbol} {worstTrade.direction}</div>
                                <div className="mt-1 text-[#d28383]">${worstTrade.pnl_usdt.toFixed(2)}</div>
                                <div className="mt-2 text-xs text-[#8b98a5]">
                                    {worstTrade.rr_achieved.toFixed(2)}R over {worstTrade.duration_minutes.toFixed(0)} min
                                </div>
                            </div>
                        ) : (
                            <div className="text-sm text-[#6b7280]">No losing trades recorded today.</div>
                        )}

                        {(data.drawdowns.periods || []).length === 0 && (
                            <div className="text-sm text-[#6b7280]">No drawdown periods recorded yet.</div>
                        )}

                        {(data.drawdowns.periods || []).slice(0, 5).map((period) => (
                            <div key={`${period.start}-${period.end}`} className="rounded-md border border-[#2d3640] p-3">
                                <div className="flex items-center justify-between gap-3">
                                    <span className="text-[#e7e9ea] text-sm font-medium">{period.depth_pct.toFixed(2)}%</span>
                                    <span className="text-xs text-[#6b7280]">{period.duration_trades} trades</span>
                                </div>
                                <div className="mt-2 text-xs text-[#8b98a5]">
                                    {new Date(period.start).toLocaleDateString('en-GB')} to {new Date(period.end).toLocaleDateString('en-GB')}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Win Rate by Symbol</span>
                    </div>
                    <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={symbolChart}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2d3640" />
                                <XAxis dataKey="symbol" stroke="#6b7280" />
                                <YAxis stroke="#6b7280" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#16181d', border: '1px solid #2d3640' }}
                                    formatter={(value: number) => [`${Number(value).toFixed(1)}%`, 'Win Rate']}
                                />
                                <Bar dataKey="win_rate_pct" fill="#4a9268" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Strategy Breakdown</span>
                    </div>
                    <div className="max-h-72 overflow-auto">
                        <table className="min-w-full text-sm">
                            <thead>
                                <tr className="text-left text-[#6b7280] border-b border-[#2d3640]">
                                    <th className="py-2">Strategy</th>
                                    <th className="py-2">Trades</th>
                                    <th className="py-2">Win Rate</th>
                                    <th className="py-2">PF</th>
                                    <th className="py-2">PnL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.strategies.strategies.map((row) => (
                                    <tr key={row.strategy} className="border-b border-[#1a1f26]">
                                        <td className="py-2 pr-2 text-[#e7e9ea]">{row.strategy}</td>
                                        <td className="py-2 text-[#8b98a5]">{row.trades}</td>
                                        <td className="py-2 text-[#8b98a5]">{(row.win_rate * 100).toFixed(1)}%</td>
                                        <td className="py-2 text-[#8b98a5]">{row.profit_factor.toFixed(2)}</td>
                                        <td className={`py-2 ${row.total_pnl_usdt >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]'}`}>
                                            {row.total_pnl_usdt >= 0 ? '+' : ''}${row.total_pnl_usdt.toFixed(2)}
                                        </td>
                                    </tr>
                                ))}
                                {data.strategies.strategies.length === 0 && (
                                    <tr>
                                        <td className="py-4 text-[#6b7280]" colSpan={5}>No strategy history available.</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div className="card">
                <div className="card-header">
                    <span className="card-title">Win Rate by Hour Heatmap</span>
                </div>
                <div className="space-y-2 overflow-x-auto">
                    <div className="grid grid-cols-[64px_repeat(24,minmax(28px,1fr))] gap-1 text-[10px] text-[#6b7280]">
                        <div />
                        {Array.from({ length: 24 }, (_, hour) => (
                            <div key={`hour-${hour}`} className="text-center">
                                {hour.toString().padStart(2, '0')}
                            </div>
                        ))}
                    </div>

                    {(data.hourly.heatmap || []).map((row) => (
                        <div key={row.day} className="grid grid-cols-[64px_repeat(24,minmax(28px,1fr))] gap-1">
                            <div className="flex items-center text-xs text-[#8b98a5]">{row.day}</div>
                            {row.hours.map((slot) => {
                                const intensity = slot.trades === 0 ? 0 : Math.max(slot.win_rate, 0.1);
                                const background =
                                    slot.trades === 0
                                        ? 'bg-[#1a1f26] text-[#4b5563]'
                                        : slot.win_rate >= 0.7
                                            ? `bg-[rgba(74,146,104,${Math.min(intensity + 0.15, 0.95)})] text-white`
                                            : slot.win_rate >= 0.4
                                                ? `bg-[rgba(201,162,39,${Math.min(intensity + 0.15, 0.85)})] text-[#0f1318]`
                                                : `bg-[rgba(166,84,84,${Math.min(0.35 + (1 - slot.win_rate), 0.95)})] text-white`;

                                return (
                                    <div
                                        key={`${row.day}-${slot.hour}`}
                                        className={`h-8 rounded flex items-center justify-center text-[10px] ${background}`}
                                        title={`${row.day} ${formatHour(slot.hour)} | ${(slot.win_rate * 100).toFixed(0)}% win | ${slot.trades} trades`}
                                    >
                                        {slot.trades}
                                    </div>
                                );
                            })}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
