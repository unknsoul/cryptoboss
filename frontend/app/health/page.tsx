'use client';

import useSWR from 'swr';

import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface LatencyPayload {
    api_roundtrip_ms?: number | null;
    last_api_request_at?: string | null;
    last_websocket_heartbeat_at?: string | null;
    websocket_heartbeat_age_ms?: number | null;
    last_binance_data_at?: string | null;
    binance_data_age_ms?: number | null;
    active_root_ws_clients?: number;
    active_price_ws_clients?: number;
}

interface ComponentHealth {
    component: string;
    status: 'GREEN' | 'AMBER' | 'RED' | string;
    detail: string;
}

interface StartupItem {
    step: string;
    completed: boolean;
    detail: string;
}

interface TimeframeFreshness {
    timeframe: string;
    last_update?: string | null;
    age_ms?: number | null;
    status: 'GREEN' | 'AMBER' | 'RED' | string;
}

interface SymbolFreshness {
    symbol: string;
    last_update?: string | null;
    age_ms?: number | null;
    status: 'GREEN' | 'AMBER' | 'RED' | string;
    timeframes: TimeframeFreshness[];
}

interface SystemSnapshot {
    engine_status?: string;
    connection_status?: string;
    exchange_health_stage?: string;
    component_health?: ComponentHealth[];
    startup_checklist?: StartupItem[];
    data_freshness?: {
        symbols: SymbolFreshness[];
        latest_market_data_at?: string | null;
        stale_symbols?: string[];
    };
}

async function fetcher<T>(url: string): Promise<T> {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }
    const payload = await response.json();
    return unwrapApiData<T>(payload);
}

function toneForStatus(status?: string): string {
    if (status === 'GREEN') {
        return 'bg-[#4a9268]/15 text-[#73c08d] border-[#335e45]';
    }
    if (status === 'AMBER' || status === 'WARNING') {
        return 'bg-[#c4a052]/15 text-[#e2c77a] border-[#5a4720]';
    }
    return 'bg-[#a65454]/15 text-[#e29696] border-[#5a2d2d]';
}

function formatAge(ageMs?: number | null): string {
    if (ageMs === null || ageMs === undefined) {
        return '--';
    }
    if (ageMs < 1000) {
        return `${ageMs} ms`;
    }
    return `${(ageMs / 1000).toFixed(1)} s`;
}

function MetricCard({
    label,
    value,
    caption,
}: {
    label: string;
    value: string;
    caption: string;
}) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">{label}</span>
            </div>
            <div className="text-2xl font-semibold text-[#e7e9ea]">{value}</div>
            <div className="mt-1 text-xs text-[#6b7280]">{caption}</div>
        </div>
    );
}

export default function HealthPage() {
    const { data: system, error: systemError } = useSWR<SystemSnapshot>(
        `${API_URL}/api/system`,
        fetcher,
        { refreshInterval: 10000, revalidateOnFocus: false }
    );
    const { data: latency, error: latencyError } = useSWR<LatencyPayload>(
        `${API_URL}/api/health/latency`,
        fetcher,
        { refreshInterval: 10000, revalidateOnFocus: false }
    );

    const exchangeStage = system?.exchange_health_stage || 'UNKNOWN';
    const componentHealth = system?.component_health ?? [];
    const freshnessRows = system?.data_freshness?.symbols ?? [];
    const startupChecklist = system?.startup_checklist ?? [];
    const staleSymbols = system?.data_freshness?.stale_symbols ?? [];

    return (
        <div className="space-y-6">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <h1 className="heading-lg mb-1">System Health</h1>
                    <p className="text-[#8b98a5] text-sm">
                        Component health, latency, and market-data freshness in one place.
                    </p>
                </div>
                <div className={`rounded-md border px-4 py-2 text-sm font-medium ${toneForStatus(exchangeStage === 'NORMAL' ? 'GREEN' : exchangeStage === 'DEGRADED' ? 'AMBER' : 'RED')}`}>
                    {exchangeStage.replace('_', ' ')}
                </div>
            </div>

            {(systemError || latencyError) && (
                <div className="card border border-[#a65454] text-[#d28383] text-sm">
                    Failed to load health telemetry.
                </div>
            )}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3 xl:grid-cols-5">
                <MetricCard
                    label="API Roundtrip"
                    value={latency?.api_roundtrip_ms != null ? `${latency.api_roundtrip_ms.toFixed(2)} ms` : '--'}
                    caption={latency?.last_api_request_at ? `Updated ${new Date(latency.last_api_request_at).toLocaleTimeString()}` : 'Waiting for request timing'}
                />
                <MetricCard
                    label="WS Heartbeat"
                    value={formatAge(latency?.websocket_heartbeat_age_ms)}
                    caption={latency?.last_websocket_heartbeat_at ? new Date(latency.last_websocket_heartbeat_at).toLocaleTimeString() : 'No heartbeat yet'}
                />
                <MetricCard
                    label="Binance Data Age"
                    value={formatAge(latency?.binance_data_age_ms)}
                    caption={latency?.last_binance_data_at ? new Date(latency.last_binance_data_at).toLocaleTimeString() : 'No live market data yet'}
                />
                <MetricCard
                    label="Root WS Clients"
                    value={`${latency?.active_root_ws_clients ?? 0}`}
                    caption={`Engine ${system?.engine_status || 'stopped'}`}
                />
                <MetricCard
                    label="Price WS Clients"
                    value={`${latency?.active_price_ws_clients ?? 0}`}
                    caption={`Connection ${system?.connection_status || 'disconnected'}`}
                />
            </div>

            <div className="card">
                <div className="card-header">
                    <span className="card-title">Component Health Matrix</span>
                    <span className="badge badge-neutral">{componentHealth.length}</span>
                </div>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {componentHealth.map((component) => (
                        <div key={component.component} className={`rounded-md border p-4 ${toneForStatus(component.status)}`}>
                            <div className="flex items-center justify-between gap-3">
                                <span className="font-medium">{component.component}</span>
                                <span className="text-xs font-semibold">{component.status}</span>
                            </div>
                            <div className="mt-2 text-sm opacity-90">{component.detail}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,0.9fr)]">
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Data Freshness</span>
                        <span className={`badge ${staleSymbols.length === 0 ? 'badge-success' : 'badge-danger'}`}>
                            {staleSymbols.length === 0 ? 'Fresh' : `${staleSymbols.length} stale`}
                        </span>
                    </div>
                    <div className="space-y-3">
                        {freshnessRows.map((row) => (
                            <div key={row.symbol} className="rounded-md border border-[#2d3640] p-4">
                                <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                                    <div>
                                        <div className="text-sm font-medium text-[#e7e9ea]">{row.symbol}</div>
                                        <div className="text-xs text-[#6b7280]">
                                            Last update {row.last_update ? new Date(row.last_update).toLocaleTimeString() : '--'}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className={`rounded-md border px-2 py-1 text-xs font-medium ${toneForStatus(row.status)}`}>
                                            {formatAge(row.age_ms)}
                                        </span>
                                    </div>
                                </div>
                                <div className="mt-3 flex flex-wrap gap-2">
                                    {row.timeframes.map((timeframe) => (
                                        <div
                                            key={`${row.symbol}-${timeframe.timeframe}`}
                                            className={`rounded-md border px-2.5 py-1.5 text-xs ${toneForStatus(timeframe.status)}`}
                                        >
                                            <span className="font-medium">{timeframe.timeframe}</span>{' '}
                                            <span className="opacity-90">{formatAge(timeframe.age_ms)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Startup Checklist</span>
                    </div>
                    <div className="space-y-3">
                        {startupChecklist.map((item) => (
                            <div key={item.step} className="rounded-md border border-[#2d3640] p-4">
                                <div className="flex items-center justify-between gap-3">
                                    <span className="text-sm font-medium text-[#e7e9ea]">{item.step}</span>
                                    <span className={`badge ${item.completed ? 'badge-success' : 'badge-warning'}`}>
                                        {item.completed ? 'DONE' : 'PENDING'}
                                    </span>
                                </div>
                                <div className="mt-2 text-sm text-[#8b98a5]">{item.detail}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
