'use client';

import { useEffect, useMemo, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface OrderBlockItem {
    id: string;
    type: 'bullish' | 'bearish';
    status: string;
    top: number;
    bottom: number;
    strength: number;
    touches: number;
}

interface FVGItem {
    id: string;
    type: 'bullish' | 'bearish';
    status: string;
    top: number;
    bottom: number;
    size: number;
    fill_pct: number;
    is_institutional: boolean;
}

interface StructureItem {
    id: string;
    type: string;
    direction: 'bullish' | 'bearish';
    break_price: number;
    broken_level: number;
    timestamp: string;
}

interface LiquidityItem {
    id: string;
    type: 'buyside' | 'sellside';
    status: string;
    price: number;
    strength: number;
    touches: number;
}

interface SetupItem {
    id: string;
    type: string;
    direction: 'long' | 'short';
    timeframe: string;
    confidence: number;
    rr: number;
    entry: number;
    sl: number;
    tp1: number;
    components: Record<string, unknown>;
    notes: string;
}

interface TFState {
    order_blocks: OrderBlockItem[];
    fvgs: FVGItem[];
    structure: StructureItem[];
    liquidity: LiquidityItem[];
    trend: string;
}

interface SMCStateResponse {
    symbol: string;
    timestamp: string;
    smc_state: Record<string, TFState>;
    setups: SetupItem[];
}

const KILL_ZONES = [
    { name: 'Asia Open', start: 0, end: 3 },
    { name: 'London Open', start: 7, end: 9 },
    { name: 'NY Open', start: 12, end: 14 },
    { name: 'London/NY Overlap', start: 13, end: 16 },
];

function SessionTimer() {
    const [clock, setClock] = useState(() => new Date());

    useEffect(() => {
        const interval = setInterval(() => setClock(new Date()), 1000);
        return () => clearInterval(interval);
    }, []);

    const nowHour = Number(clock.toISOString().slice(11, 13));
    const inZone = KILL_ZONES.find((zone) => nowHour >= zone.start && nowHour < zone.end);

    const nextZone = KILL_ZONES
        .map((zone) => {
            const nowMinutes = nowHour * 60 + clock.getUTCMinutes();
            let startMinutes = zone.start * 60;
            if (startMinutes <= nowMinutes) {
                startMinutes += 24 * 60;
            }
            return { zone, minutesAway: startMinutes - nowMinutes };
        })
        .sort((a, b) => a.minutesAway - b.minutesAway)[0];

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">SessionTimer</span>
            </div>
            <div className="space-y-2">
                <div className="text-sm text-[#8b98a5]">UTC {clock.toISOString().slice(11, 19)}</div>
                {inZone ? (
                    <div className="badge badge-warning">Active Kill Zone: {inZone.name}</div>
                ) : (
                    <div className="badge badge-neutral">No kill zone active</div>
                )}
                <div className="text-xs text-[#6b7280]">
                    Next: {nextZone.zone.name} in {nextZone.minutesAway}m
                </div>
            </div>
        </div>
    );
}

function ConfluenceScorer({ setups }: { setups: SetupItem[] }) {
    const score = useMemo(() => {
        if (!setups.length) {
            return 0;
        }
        return Math.round((setups[0].confidence || 0) * 100);
    }, [setups]);

    const tone = score >= 75 ? 'bg-[#4a9268]' : score >= 55 ? 'bg-[#c4a052]' : 'bg-[#a65454]';

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">ConfluenceScorer</span>
            </div>
            <div className="space-y-2">
                <div className="text-3xl font-semibold text-white">{score}</div>
                <div className="h-2 rounded-full bg-[#1a1f26] overflow-hidden">
                    <div className={`h-full ${tone}`} style={{ width: `${score}%` }} />
                </div>
                <div className="text-xs text-[#6b7280]">Confluence meter 0-100</div>
            </div>
        </div>
    );
}

function MTFAlignmentGrid({ smcState }: { smcState: Record<string, TFState> }) {
    const timeframes = ['1m', '5m', '15m', '1h', '4h'];

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">MTFAlignmentGrid</span>
            </div>
            <div className="grid grid-cols-5 gap-2">
                {timeframes.map((tf) => {
                    const trend = smcState[tf]?.trend || 'unknown';
                    const tone = trend === 'bullish' ? 'badge-success' : trend === 'bearish' ? 'badge-danger' : 'badge-neutral';
                    return (
                        <div key={tf} className="border border-[#2d3640] rounded-md p-2 text-center">
                            <div className="text-xs text-[#6b7280] mb-1">{tf}</div>
                            <span className={`badge ${tone}`}>{trend}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function SMCStatePanel({ blocks }: { blocks: OrderBlockItem[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">SMCStatePanel - Order Blocks</span>
            </div>
            <div className="space-y-2 max-h-72 overflow-auto pr-1">
                {blocks.length === 0 && <div className="text-sm text-[#6b7280]">No order blocks</div>}
                {blocks.slice(0, 12).map((block) => (
                    <div key={block.id} className="border border-[#2d3640] rounded-md p-2">
                        <div className="flex items-center justify-between">
                            <span className={`badge ${block.type === 'bullish' ? 'badge-success' : 'badge-danger'}`}>{block.type}</span>
                            <span className="text-xs text-[#8b98a5]">{block.status}</span>
                        </div>
                        <div className="text-xs text-[#8b98a5] mt-2">
                            Zone {block.bottom.toFixed(2)} - {block.top.toFixed(2)}
                        </div>
                        <div className="text-xs text-[#6b7280]">
                            Strength {(block.strength * 100).toFixed(0)}% | Touches {block.touches}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

function FVGPanel({ fvgs }: { fvgs: FVGItem[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">FVGPanel</span>
            </div>
            <div className="space-y-2 max-h-72 overflow-auto pr-1">
                {fvgs.length === 0 && <div className="text-sm text-[#6b7280]">No FVGs</div>}
                {fvgs.slice(0, 12).map((fvg) => (
                    <div key={fvg.id} className="border border-[#2d3640] rounded-md p-2">
                        <div className="flex items-center justify-between">
                            <span className={`badge ${fvg.type === 'bullish' ? 'badge-success' : 'badge-danger'}`}>{fvg.type}</span>
                            <span className="text-xs text-[#8b98a5]">{fvg.status}</span>
                        </div>
                        <div className="text-xs text-[#8b98a5] mt-2">
                            Size {fvg.size.toFixed(2)} | Fill {fvg.fill_pct.toFixed(1)}%
                        </div>
                        {fvg.is_institutional && <div className="text-xs text-[#c4a052] mt-1">Institutional gap</div>}
                    </div>
                ))}
            </div>
        </div>
    );
}

function StructurePanel({ structure }: { structure: StructureItem[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">StructurePanel - BOS/CHoCH Timeline</span>
            </div>
            <div className="space-y-2 max-h-72 overflow-auto pr-1">
                {structure.length === 0 && <div className="text-sm text-[#6b7280]">No structure events</div>}
                {structure.slice(0, 12).map((event) => (
                    <div key={event.id} className="flex items-center justify-between border-b border-[#2d3640] pb-2">
                        <div>
                            <div className={`text-sm ${event.type.includes('choch') ? 'text-[#c4a052]' : 'text-[#8b98a5]'}`}>{event.type}</div>
                            <div className="text-xs text-[#6b7280]">{new Date(event.timestamp).toLocaleString()}</div>
                        </div>
                        <div className={`badge ${event.direction === 'bullish' ? 'badge-success' : 'badge-danger'}`}>{event.direction}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}

function LiquidityPanel({ liquidity }: { liquidity: LiquidityItem[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">LiquidityPanel</span>
            </div>
            <div className="space-y-2 max-h-72 overflow-auto pr-1">
                {liquidity.length === 0 && <div className="text-sm text-[#6b7280]">No liquidity levels</div>}
                {liquidity.slice(0, 12).map((level) => (
                    <div key={level.id} className="border border-[#2d3640] rounded-md p-2">
                        <div className="flex items-center justify-between">
                            <span className={`badge ${level.type === 'buyside' ? 'badge-danger' : 'badge-success'}`}>{level.type}</span>
                            <span className="text-xs text-[#8b98a5]">{level.status}</span>
                        </div>
                        <div className="text-xs text-[#8b98a5] mt-2">
                            Price {level.price.toFixed(2)} | Touches {level.touches}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

function SetupFeed({ setups }: { setups: SetupItem[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">SetupFeed</span>
            </div>
            <div className="space-y-2 max-h-96 overflow-auto pr-1">
                {setups.length === 0 && <div className="text-sm text-[#6b7280]">No high-confidence setups</div>}
                {setups.slice(0, 10).map((setup) => (
                    <div key={setup.id} className="border border-[#2d3640] rounded-md p-3">
                        <div className="flex items-center justify-between">
                            <span className={`badge ${setup.direction === 'long' ? 'badge-success' : 'badge-danger'}`}>{setup.direction}</span>
                            <span className="text-xs text-[#8b98a5]">{setup.timeframe}</span>
                        </div>
                        <div className="text-sm text-white mt-2">{setup.type}</div>
                        <div className="text-xs text-[#8b98a5] mt-1">
                            Entry {setup.entry.toFixed(2)} | SL {setup.sl.toFixed(2)} | TP1 {setup.tp1.toFixed(2)}
                        </div>
                        <div className="text-xs text-[#6b7280] mt-1">
                            RR {setup.rr.toFixed(2)} | Confidence {(setup.confidence * 100).toFixed(0)}%
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default function SMCPage() {
    const [symbol, setSymbol] = useState('BTC/USDT');
    const [timeframe, setTimeframe] = useState('5m');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [payload, setPayload] = useState<SMCStateResponse | null>(null);

    useEffect(() => {
        let active = true;

        const fetchState = async () => {
            try {
                const response = await fetch(
                    `${API_URL}/api/v2/smc/state?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}`,
                );
                if (!response.ok) {
                    throw new Error('Failed to fetch SMC state');
                }
                const result = await response.json();
                if (!active) {
                    return;
                }
                setPayload((result.data || result) as SMCStateResponse);
                setError(null);
            } catch (err: unknown) {
                const message = err instanceof Error ? err.message : 'Unknown error';
                if (active) {
                    setError(message);
                    setPayload(null);
                }
            } finally {
                if (active) {
                    setLoading(false);
                }
            }
        };

        setLoading(true);
        fetchState();
        const interval = setInterval(fetchState, 15000);

        return () => {
            active = false;
            clearInterval(interval);
        };
    }, [symbol, timeframe]);

    const tfState = payload?.smc_state?.[timeframe];
    const orderBlocks = tfState?.order_blocks || [];
    const fvgs = tfState?.fvgs || [];
    const structure = tfState?.structure || [];
    const liquidity = tfState?.liquidity || [];

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                    <h1 className="heading-lg">SMC Dashboard</h1>
                    <p className="text-sm text-[#8b98a5]">Live Smart Money Concepts state and confluence setups</p>
                </div>
                <div className="flex items-center gap-2">
                    <select
                        className="bg-[#1a1f26] border border-[#2d3640] rounded px-3 py-2 text-sm"
                        value={symbol}
                        onChange={(event) => setSymbol(event.target.value)}
                    >
                        <option>BTC/USDT</option>
                        <option>ETH/USDT</option>
                        <option>SOL/USDT</option>
                        <option>BNB/USDT</option>
                    </select>
                    <select
                        className="bg-[#1a1f26] border border-[#2d3640] rounded px-3 py-2 text-sm"
                        value={timeframe}
                        onChange={(event) => setTimeframe(event.target.value)}
                    >
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                        <option value="4h">4h</option>
                    </select>
                </div>
            </div>

            {loading && <div className="card text-sm text-[#8b98a5]">Loading SMC state...</div>}
            {error && <div className="card text-sm text-[#a65454]">{error}</div>}

            {!loading && !error && payload && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <SessionTimer />
                        <ConfluenceScorer setups={payload.setups} />
                        <MTFAlignmentGrid smcState={payload.smc_state} />
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <SMCStatePanel blocks={orderBlocks} />
                        <FVGPanel fvgs={fvgs} />
                        <StructurePanel structure={structure} />
                        <LiquidityPanel liquidity={liquidity} />
                    </div>

                    <SetupFeed setups={payload.setups} />
                </>
            )}
        </div>
    );
}
