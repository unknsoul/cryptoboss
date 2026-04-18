'use client';

import { useEffect, useMemo, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ScalperSignal {
    action: string;
    reason: string;
    confidence: number;
    size: number;
    price: number;
    stop_loss?: number;
    take_profit?: number;
    metadata?: Record<string, unknown>;
}

interface TopSetup {
    id: string;
    type: string;
    direction: 'long' | 'short';
    timeframe: string;
    confidence: number;
    rr: number;
    entry: number;
    sl: number;
    tp1: number;
    components: string[];
}

interface ScalperAnalysis {
    session: string | null;
    in_kill_zone: boolean;
    kill_zone_name: string;
    setups_count: number;
    top_setups: TopSetup[];
}

interface ScalperPayload {
    symbol: string;
    signal: ScalperSignal;
    analysis: ScalperAnalysis;
}

function ScalperControlPanel({
    running,
    setRunning,
    sessionFilter,
    setSessionFilter,
    killZoneOnly,
    setKillZoneOnly,
}: {
    running: boolean;
    setRunning: (value: boolean) => void;
    sessionFilter: boolean;
    setSessionFilter: (value: boolean) => void;
    killZoneOnly: boolean;
    setKillZoneOnly: (value: boolean) => void;
}) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">ScalperControlPanel</span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
                <button
                    className={`px-3 py-2 rounded text-sm font-medium ${running ? 'bg-[#a65454] text-white' : 'bg-[#4a9268] text-white'}`}
                    onClick={() => setRunning(!running)}
                >
                    {running ? 'Stop Scalper' : 'Start Scalper'}
                </button>
                <label className="flex items-center gap-2 text-sm text-[#8b98a5]">
                    <input
                        type="checkbox"
                        checked={sessionFilter}
                        onChange={(event) => setSessionFilter(event.target.checked)}
                    />
                    Session filter
                </label>
                <label className="flex items-center gap-2 text-sm text-[#8b98a5]">
                    <input
                        type="checkbox"
                        checked={killZoneOnly}
                        onChange={(event) => setKillZoneOnly(event.target.checked)}
                    />
                    Kill zone only
                </label>
            </div>
        </div>
    );
}

function ActiveSetupsTable({ setups }: { setups: TopSetup[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">ActiveSetupsTable</span>
            </div>
            <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                    <thead>
                        <tr className="text-left text-[#6b7280] border-b border-[#2d3640]">
                            <th className="py-2">Type</th>
                            <th className="py-2">Direction</th>
                            <th className="py-2">RR</th>
                            <th className="py-2">Confidence</th>
                            <th className="py-2">Entry</th>
                            <th className="py-2">SL</th>
                            <th className="py-2">TP1</th>
                        </tr>
                    </thead>
                    <tbody>
                        {setups.length === 0 && (
                            <tr>
                                <td className="py-4 text-[#6b7280]" colSpan={7}>
                                    No active setups
                                </td>
                            </tr>
                        )}
                        {setups.map((setup) => (
                            <tr key={setup.id} className="border-b border-[#1a1f26]">
                                <td className="py-2 text-[#e7e9ea]">{setup.type}</td>
                                <td className="py-2">
                                    <span className={`badge ${setup.direction === 'long' ? 'badge-success' : 'badge-danger'}`}>
                                        {setup.direction}
                                    </span>
                                </td>
                                <td className="py-2 text-[#8b98a5]">{setup.rr.toFixed(2)}</td>
                                <td className="py-2 text-[#8b98a5]">{(setup.confidence * 100).toFixed(0)}%</td>
                                <td className="py-2 text-[#8b98a5]">{setup.entry.toFixed(2)}</td>
                                <td className="py-2 text-[#8b98a5]">{setup.sl.toFixed(2)}</td>
                                <td className="py-2 text-[#8b98a5]">{setup.tp1.toFixed(2)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function ScalperPnL({ signal }: { signal: ScalperSignal | null }) {
    const estimatedR = useMemo(() => {
        if (!signal?.price || !signal?.stop_loss || !signal?.take_profit) {
            return 0;
        }
        const risk = Math.abs(signal.price - signal.stop_loss);
        if (risk <= 0) {
            return 0;
        }
        return Math.abs(signal.take_profit - signal.price) / risk;
    }, [signal]);

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">ScalperP&L</span>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <div className="text-[#6b7280]">Action</div>
                    <div className="text-[#e7e9ea]">{signal?.action || 'HOLD'}</div>
                </div>
                <div>
                    <div className="text-[#6b7280]">Estimated R</div>
                    <div className="text-[#e7e9ea]">{estimatedR.toFixed(2)}</div>
                </div>
                <div>
                    <div className="text-[#6b7280]">Entry</div>
                    <div className="text-[#8b98a5]">{signal?.price ? signal.price.toFixed(2) : '--'}</div>
                </div>
                <div>
                    <div className="text-[#6b7280]">Position Size</div>
                    <div className="text-[#8b98a5]">{signal?.size ? signal.size.toFixed(6) : '--'}</div>
                </div>
            </div>
        </div>
    );
}

function SessionHeatmap({ session }: { session: string | null }) {
    const hours = Array.from({ length: 24 }, (_, index) => index);

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">SessionHeatmap</span>
            </div>
            <div className="grid grid-cols-12 gap-1">
                {hours.map((hour) => {
                    const london = hour >= 7 && hour < 16;
                    const ny = hour >= 13 && hour < 22;
                    const active = london || ny;
                    const highlight = session && ((session === 'London' && london) || (session === 'NY' && ny) || (session === 'Overlap' && london && ny));
                    return (
                        <div
                            key={hour}
                            className={`h-6 rounded text-[10px] flex items-center justify-center ${highlight
                                ? 'bg-[#c4a052] text-[#0f1419]'
                                : active
                                    ? 'bg-[#5b7a9d]/30 text-[#8b98a5]'
                                    : 'bg-[#1a1f26] text-[#4a5568]'
                                }`}
                        >
                            {hour}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function KillZoneAlert({ active, name }: { active: boolean; name: string }) {
    return (
        <div className={`card border ${active ? 'border-[#c4a052]' : 'border-[#2d3640]'}`}>
            <div className="card-header">
                <span className="card-title">KillZoneAlert</span>
            </div>
            {active ? (
                <div className="bg-[#c4a052]/20 border border-[#c4a052]/40 rounded-md px-3 py-2 text-sm text-[#e7e9ea] animate-pulse">
                    Active kill zone: {name || 'current window'}
                </div>
            ) : (
                <div className="text-sm text-[#6b7280]">No active kill zone</div>
            )}
        </div>
    );
}

function Riskometer({ exposurePct }: { exposurePct: number }) {
    const pct = Math.max(0, Math.min(exposurePct, 100));
    const tone = pct < 20 ? 'bg-[#4a9268]' : pct < 40 ? 'bg-[#c4a052]' : 'bg-[#a65454]';

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">Riskometer</span>
            </div>
            <div className="space-y-2">
                <div className="text-2xl font-semibold">{pct.toFixed(1)}%</div>
                <div className="h-2 rounded-full bg-[#1a1f26] overflow-hidden">
                    <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
                </div>
                <div className="text-xs text-[#6b7280]">Current exposure as % of account balance</div>
            </div>
        </div>
    );
}

export default function ScalperPage() {
    const [running, setRunning] = useState(false);
    const [sessionFilter, setSessionFilter] = useState(true);
    const [killZoneOnly, setKillZoneOnly] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [payload, setPayload] = useState<ScalperPayload | null>(null);

    useEffect(() => {
        let active = true;

        const fetchLive = async () => {
            if (!running) {
                setLoading(false);
                return;
            }

            try {
                const response = await fetch(`${API_URL}/api/v2/scalper/live?symbol=BTC%2FUSDT`);
                if (!response.ok) {
                    throw new Error('Failed to fetch scalper data');
                }
                const result = await response.json();
                if (!active) {
                    return;
                }
                setPayload((result.data || result) as ScalperPayload);
                setError(null);
            } catch (err: unknown) {
                const message = err instanceof Error ? err.message : 'Unknown error';
                if (active) {
                    setError(message);
                }
            } finally {
                if (active) {
                    setLoading(false);
                }
            }
        };

        setLoading(true);
        fetchLive();
        const interval = setInterval(fetchLive, 10000);

        return () => {
            active = false;
            clearInterval(interval);
        };
    }, [running, sessionFilter, killZoneOnly]);

    const exposurePct = useMemo(() => {
        if (!payload?.signal?.size || !payload.signal.price) {
            return 0;
        }
        const notion = payload.signal.size * payload.signal.price;
        return (notion / 10000) * 100;
    }, [payload]);

    return (
        <div className="space-y-6">
            <div>
                <h1 className="heading-lg">Intraday Scalper Live View</h1>
                <p className="text-sm text-[#8b98a5]">Session-aware SMC scalping dashboard</p>
            </div>

            <ScalperControlPanel
                running={running}
                setRunning={setRunning}
                sessionFilter={sessionFilter}
                setSessionFilter={setSessionFilter}
                killZoneOnly={killZoneOnly}
                setKillZoneOnly={setKillZoneOnly}
            />

            {error && <div className="card text-sm text-[#a65454]">{error}</div>}
            {!running && <div className="card text-sm text-[#8b98a5]">Scalper is stopped. Press Start Scalper to begin polling live setups.</div>}
            {loading && running && <div className="card text-sm text-[#8b98a5]">Loading live scalper data...</div>}

            {payload && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <ScalperPnL signal={payload.signal} />
                        <KillZoneAlert active={payload.analysis.in_kill_zone} name={payload.analysis.kill_zone_name} />
                        <Riskometer exposurePct={exposurePct} />
                    </div>

                    <SessionHeatmap session={payload.analysis.session} />
                    <ActiveSetupsTable setups={payload.analysis.top_setups} />
                </>
            )}
        </div>
    );
}
