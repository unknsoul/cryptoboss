'use client';

import { useEffect, useMemo, useState } from 'react';
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
    Bar,
    BarChart,
} from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface PresetDetails {
    name: string;
    description: string;
    tags: string[];
}

interface StrategyItem {
    id: string;
    name: string;
    symbol: string;
    timeframe: string;
    tags: string[];
}

interface BacktestSummary {
    strategy_name: string;
    net_profit_usd: number;
    net_profit_pct: number;
    total_trades: number;
    win_rate_pct: number;
    profit_factor: number;
    max_drawdown_pct: number;
    sharpe_ratio: number;
    avg_rr: number;
}

interface TradeLogItem {
    id: string;
    direction: string;
    entry_time: string;
    exit_time: string;
    entry: number;
    exit: number;
    net_pnl: number;
    pnl_pct: number;
    exit_reason: string;
    rr: number;
}

interface BacktestResult {
    summary: BacktestSummary;
    trades: TradeLogItem[];
    equity_curve: number[];
    drawdown: number[];
    monthly_returns: Record<string, number>;
    monte_carlo: Record<string, number | string>;
}

interface WalkForwardResult {
    walk_forward: {
        n_splits: number;
        folds: Array<{
            fold: number;
            train_pf: number;
            oos_profit_factor: number;
            oos_win_rate: number;
            oos_net_profit: number;
            oos_sharpe: number;
        }>;
    };
}

function PresetSelector({
    presets,
    selected,
    setSelected,
    onLoad,
}: {
    presets: string[];
    selected: string;
    setSelected: (value: string) => void;
    onLoad: () => void;
}) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">PresetSelector</span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
                <select
                    className="bg-[#1a1f26] border border-[#2d3640] rounded px-3 py-2 text-sm"
                    value={selected}
                    onChange={(event) => setSelected(event.target.value)}
                >
                    {presets.map((preset) => (
                        <option key={preset} value={preset}>
                            {preset}
                        </option>
                    ))}
                </select>
                <button className="btn btn-primary" onClick={onLoad}>
                    Load Preset
                </button>
            </div>
        </div>
    );
}

function StrategyCanvas({
    strategyName,
    blocks,
    onDropIndicator,
}: {
    strategyName: string | null;
    blocks: string[];
    onDropIndicator: (indicator: string) => void;
}) {
    return (
        <div className="card min-h-[180px]">
            <div className="card-header">
                <span className="card-title">StrategyCanvas</span>
            </div>
            <div
                className="h-full border border-dashed border-[#2d3640] rounded-md p-4 text-sm text-[#8b98a5]"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                    event.preventDefault();
                    const indicator = event.dataTransfer.getData('text/plain');
                    if (indicator) {
                        onDropIndicator(indicator);
                    }
                }}
            >
                {strategyName ? (
                    <>
                        <div className="text-[#e7e9ea] font-medium mb-2">{strategyName}</div>
                        <div className="mb-3">Drop indicators here to compose condition blocks.</div>
                        <div className="flex flex-wrap gap-2">
                            {blocks.length === 0 && <span className="text-xs text-[#6b7280]">No indicator blocks dropped yet.</span>}
                            {blocks.map((block, index) => (
                                <span key={`${block}-${index}`} className="badge badge-accent">
                                    {block}
                                </span>
                            ))}
                        </div>
                    </>
                ) : (
                    <div>Load a preset or create a strategy to populate the visual canvas.</div>
                )}
            </div>
        </div>
    );
}

function IndicatorLibrary({ onPickIndicator }: { onPickIndicator: (indicator: string) => void }) {
    const indicators = ['Order Block', 'FVG', 'BOS', 'CHoCH', 'EMA', 'RSI', 'VWAP', 'Volume MA'];
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">IndicatorLibrary</span>
            </div>
            <div className="flex flex-wrap gap-2">
                {indicators.map((indicator) => (
                    <span
                        key={indicator}
                        className="badge badge-neutral cursor-grab"
                        draggable
                        onDragStart={(event) => event.dataTransfer.setData('text/plain', indicator)}
                        onClick={() => onPickIndicator(indicator)}
                    >
                        {indicator}
                    </span>
                ))}
            </div>
        </div>
    );
}

function ConditionEditor({ blocks }: { blocks: string[] }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">ConditionEditor</span>
            </div>
            <div className="text-sm text-[#8b98a5] mb-2">Operators and thresholds are managed by the loaded strategy preset. Dropped blocks are listed below.</div>
            <div className="flex flex-wrap gap-2">
                {blocks.length === 0 && <span className="text-xs text-[#6b7280]">No blocks staged</span>}
                {blocks.map((block, index) => (
                    <span key={`${block}-condition-${index}`} className="badge badge-neutral">
                        {block}
                    </span>
                ))}
            </div>
        </div>
    );
}

function RiskConfigPanel({ riskPct, setRiskPct }: { riskPct: number; setRiskPct: (value: number) => void }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">RiskConfigPanel</span>
            </div>
            <label className="text-sm text-[#8b98a5] block mb-2">Risk per trade (%)</label>
            <input
                type="number"
                step="0.1"
                min="0.1"
                max="5"
                value={riskPct}
                onChange={(event) => setRiskPct(Number(event.target.value))}
                className="w-full bg-[#1a1f26] border border-[#2d3640] rounded px-3 py-2 text-sm"
            />
        </div>
    );
}

function FilterConfigPanel({ killZoneOnly, setKillZoneOnly }: { killZoneOnly: boolean; setKillZoneOnly: (value: boolean) => void }) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">FilterConfigPanel</span>
            </div>
            <label className="flex items-center gap-2 text-sm text-[#8b98a5]">
                <input type="checkbox" checked={killZoneOnly} onChange={(event) => setKillZoneOnly(event.target.checked)} />
                Kill zone only
            </label>
        </div>
    );
}

function BacktestPanel({
    hasStrategy,
    onBacktest,
    onWalkForward,
    running,
}: {
    hasStrategy: boolean;
    onBacktest: () => void;
    onWalkForward: () => void;
    running: boolean;
}) {
    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">BacktestPanel</span>
            </div>
            <div className="flex flex-wrap gap-3">
                <button className="btn btn-primary" disabled={!hasStrategy || running} onClick={onBacktest}>
                    {running ? 'Running...' : 'Run Backtest'}
                </button>
                <button className="btn btn-ghost border border-[#2d3640]" disabled={!hasStrategy || running} onClick={onWalkForward}>
                    Run Walk-Forward
                </button>
            </div>
        </div>
    );
}

function ResultsDashboard({ result }: { result: BacktestResult | null }) {
    if (!result) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title">ResultsDashboard</span>
                </div>
                <div className="text-sm text-[#6b7280]">Run a backtest to display metrics and trade log.</div>
            </div>
        );
    }

    const equityData = result.equity_curve.map((value, index) => ({ index, equity: value }));

    return (
        <div className="card space-y-4">
            <div className="card-header">
                <span className="card-title">ResultsDashboard</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="border border-[#2d3640] rounded p-2">
                    <div className="text-[#6b7280]">Net PnL</div>
                    <div className="text-[#e7e9ea]">${result.summary.net_profit_usd.toFixed(2)}</div>
                </div>
                <div className="border border-[#2d3640] rounded p-2">
                    <div className="text-[#6b7280]">Win Rate</div>
                    <div className="text-[#e7e9ea]">{result.summary.win_rate_pct.toFixed(2)}%</div>
                </div>
                <div className="border border-[#2d3640] rounded p-2">
                    <div className="text-[#6b7280]">Profit Factor</div>
                    <div className="text-[#e7e9ea]">{result.summary.profit_factor.toFixed(2)}</div>
                </div>
                <div className="border border-[#2d3640] rounded p-2">
                    <div className="text-[#6b7280]">Max Drawdown</div>
                    <div className="text-[#e7e9ea]">{result.summary.max_drawdown_pct.toFixed(2)}%</div>
                </div>
            </div>

            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={equityData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3640" />
                        <XAxis dataKey="index" stroke="#6b7280" />
                        <YAxis stroke="#6b7280" />
                        <Tooltip contentStyle={{ backgroundColor: '#16181d', border: '1px solid #2d3640' }} />
                        <Line type="monotone" dataKey="equity" stroke="#5b7a9d" dot={false} strokeWidth={2} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="max-h-56 overflow-auto">
                <table className="min-w-full text-sm">
                    <thead>
                        <tr className="text-left text-[#6b7280] border-b border-[#2d3640]">
                            <th className="py-2">ID</th>
                            <th className="py-2">Dir</th>
                            <th className="py-2">PnL</th>
                            <th className="py-2">RR</th>
                            <th className="py-2">Exit</th>
                        </tr>
                    </thead>
                    <tbody>
                        {result.trades.slice(0, 40).map((trade) => (
                            <tr key={trade.id} className="border-b border-[#1a1f26]">
                                <td className="py-2 text-[#8b98a5]">{trade.id}</td>
                                <td className="py-2 text-[#8b98a5]">{trade.direction}</td>
                                <td className={`py-2 ${trade.net_pnl >= 0 ? 'text-[#4a9268]' : 'text-[#a65454]'}`}>{trade.net_pnl.toFixed(2)}</td>
                                <td className="py-2 text-[#8b98a5]">{trade.rr.toFixed(2)}</td>
                                <td className="py-2 text-[#8b98a5]">{trade.exit_reason}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function MonteCarloChart({ result }: { result: BacktestResult | null }) {
    const data = useMemo(() => {
        if (!result) {
            return [];
        }
        return Object.entries(result.monte_carlo)
            .filter(([, value]) => typeof value === 'number')
            .map(([key, value]) => ({ key, value: Number(value) }))
            .slice(0, 8);
    }, [result]);

    return (
        <div className="card h-72">
            <div className="card-header">
                <span className="card-title">MonteCarloChart</span>
            </div>
            {data.length === 0 ? (
                <div className="text-sm text-[#6b7280]">Monte Carlo metrics appear after backtest.</div>
            ) : (
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3640" />
                        <XAxis dataKey="key" stroke="#6b7280" tick={{ fontSize: 10 }} interval={0} angle={-15} height={60} />
                        <YAxis stroke="#6b7280" />
                        <Tooltip contentStyle={{ backgroundColor: '#16181d', border: '1px solid #2d3640' }} />
                        <Bar dataKey="value" fill="#5b7a9d" />
                    </BarChart>
                </ResponsiveContainer>
            )}
        </div>
    );
}

function WalkForwardChart({ result }: { result: WalkForwardResult | null }) {
    const folds = result?.walk_forward?.folds || [];

    return (
        <div className="card h-72">
            <div className="card-header">
                <span className="card-title">WalkForwardChart</span>
            </div>
            {folds.length === 0 ? (
                <div className="text-sm text-[#6b7280]">Run walk-forward to see out-of-sample fold performance.</div>
            ) : (
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={folds}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3640" />
                        <XAxis dataKey="fold" stroke="#6b7280" />
                        <YAxis stroke="#6b7280" />
                        <Tooltip contentStyle={{ backgroundColor: '#16181d', border: '1px solid #2d3640' }} />
                        <Line dataKey="oos_profit_factor" stroke="#4a9268" strokeWidth={2} dot />
                        <Line dataKey="oos_win_rate" stroke="#c4a052" strokeWidth={2} dot />
                    </LineChart>
                </ResponsiveContainer>
            )}
        </div>
    );
}

function StrategyExportModal({
    open,
    onClose,
    json,
}: {
    open: boolean;
    onClose: () => void;
    json: string;
}) {
    if (!open) {
        return null;
    }

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-[#0f1419] border border-[#2d3640] rounded-lg w-full max-w-3xl max-h-[85vh] overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-[#2d3640]">
                    <h3 className="text-sm font-medium text-[#e7e9ea]">StrategyExportModal</h3>
                    <button className="text-[#8b98a5] hover:text-[#e7e9ea]" onClick={onClose}>Close</button>
                </div>
                <pre className="p-4 text-xs text-[#8b98a5] overflow-auto">{json}</pre>
            </div>
        </div>
    );
}

export default function BuilderPage() {
    const [presets, setPresets] = useState<string[]>([]);
    const [presetDetails, setPresetDetails] = useState<Record<string, PresetDetails>>({});
    const [selectedPreset, setSelectedPreset] = useState('smc_scalper');
    const [strategies, setStrategies] = useState<StrategyItem[]>([]);
    const [activeStrategyId, setActiveStrategyId] = useState<string | null>(null);
    const [activeStrategyJson, setActiveStrategyJson] = useState('');
    const [riskPct, setRiskPct] = useState(0.5);
    const [killZoneOnly, setKillZoneOnly] = useState(false);
    const [running, setRunning] = useState(false);
    const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
    const [walkForwardResult, setWalkForwardResult] = useState<WalkForwardResult | null>(null);
    const [showExport, setShowExport] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [canvasBlocks, setCanvasBlocks] = useState<string[]>([]);

    useEffect(() => {
        const bootstrap = async () => {
            try {
                const [presetsRes, strategiesRes] = await Promise.all([
                    fetch(`${API_URL}/api/v2/strategies/presets`),
                    fetch(`${API_URL}/api/v2/strategies`),
                ]);

                const presetsJson = await presetsRes.json();
                const strategiesJson = await strategiesRes.json();

                const presetData = presetsJson.data || presetsJson;
                const strategyData = strategiesJson.data || strategiesJson;

                setPresets(presetData.presets || []);
                setPresetDetails((presetData.details || {}) as Record<string, PresetDetails>);
                setStrategies((strategyData.strategies || []) as StrategyItem[]);
                setError(null);
            } catch (err: unknown) {
                const message = err instanceof Error ? err.message : 'Unknown error';
                setError(message);
            }
        };

        bootstrap();
    }, []);

    const activeStrategy = useMemo(() => strategies.find((item) => item.id === activeStrategyId) || null, [strategies, activeStrategyId]);

    const loadPreset = async () => {
        try {
            setRunning(true);
            const response = await fetch(
                `${API_URL}/api/v2/strategies/load-preset?preset=${encodeURIComponent(selectedPreset)}&symbol=BTC%2FUSDT&timeframe=5m`,
                { method: 'POST' },
            );
            if (!response.ok) {
                throw new Error('Failed to load preset');
            }
            const result = await response.json();
            const payload = result.data || result;
            const strategyId = payload.strategy_id as string;
            setActiveStrategyId(strategyId);
            setActiveStrategyJson(JSON.stringify(payload.strategy, null, 2));
            setCanvasBlocks([]);

            const strategiesRes = await fetch(`${API_URL}/api/v2/strategies`);
            const strategiesJson = await strategiesRes.json();
            const strategiesData = strategiesJson.data || strategiesJson;
            setStrategies((strategiesData.strategies || []) as StrategyItem[]);
            setError(null);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
        } finally {
            setRunning(false);
        }
    };

    const runBacktest = async () => {
        if (!activeStrategyId) {
            return;
        }

        try {
            setRunning(true);
            const response = await fetch(`${API_URL}/api/v2/strategies/${activeStrategyId}/backtest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ limit: 1200, n_simulations: 500, risk_pct: riskPct / 100, kill_zone_only: killZoneOnly }),
            });
            if (!response.ok) {
                throw new Error('Backtest failed');
            }
            const result = await response.json();
            setBacktestResult((result.data || result) as BacktestResult);
            setError(null);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
        } finally {
            setRunning(false);
        }
    };

    const runWalkForward = async () => {
        if (!activeStrategyId) {
            return;
        }

        try {
            setRunning(true);
            const response = await fetch(`${API_URL}/api/v2/strategies/${activeStrategyId}/walk-forward`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n_splits: 5 }),
            });
            if (!response.ok) {
                throw new Error('Walk-forward failed');
            }
            const result = await response.json();
            setWalkForwardResult((result.data || result) as WalkForwardResult);
            setError(null);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
        } finally {
            setRunning(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <h1 className="heading-lg">Visual Strategy Builder + Tester</h1>
                    <p className="text-sm text-[#8b98a5]">Compose, backtest, and validate custom strategies with institutional analytics</p>
                </div>
                <button className="btn btn-ghost border border-[#2d3640]" disabled={!activeStrategyJson} onClick={() => setShowExport(true)}>
                    Export Strategy JSON
                </button>
            </div>

            {error && <div className="card text-sm text-[#a65454]">{error}</div>}

            <PresetSelector
                presets={presets}
                selected={selectedPreset}
                setSelected={setSelectedPreset}
                onLoad={loadPreset}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="lg:col-span-2 space-y-4">
                    <StrategyCanvas
                        strategyName={activeStrategy?.name || null}
                        blocks={canvasBlocks}
                        onDropIndicator={(indicator) => setCanvasBlocks((prev) => [...prev, indicator])}
                    />
                    <ConditionEditor blocks={canvasBlocks} />
                </div>
                <div className="space-y-4">
                    <IndicatorLibrary onPickIndicator={(indicator) => setCanvasBlocks((prev) => [...prev, indicator])} />
                    <RiskConfigPanel riskPct={riskPct} setRiskPct={setRiskPct} />
                    <FilterConfigPanel killZoneOnly={killZoneOnly} setKillZoneOnly={setKillZoneOnly} />
                </div>
            </div>

            <BacktestPanel hasStrategy={Boolean(activeStrategyId)} onBacktest={runBacktest} onWalkForward={runWalkForward} running={running} />

            <ResultsDashboard result={backtestResult} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <MonteCarloChart result={backtestResult} />
                <WalkForwardChart result={walkForwardResult} />
            </div>

            <StrategyExportModal open={showExport} onClose={() => setShowExport(false)} json={activeStrategyJson} />

            {selectedPreset in presetDetails && (
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Loaded Preset</span>
                    </div>
                    <div className="text-sm text-[#8b98a5]">{presetDetails[selectedPreset].description}</div>
                    <div className="flex flex-wrap gap-2 mt-3">
                        {presetDetails[selectedPreset].tags?.map((tag) => (
                            <span key={tag} className="badge badge-neutral">{tag}</span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
