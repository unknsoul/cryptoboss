'use client';

import type { ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useSWR from 'swr';

import { unwrapApiData } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const;

type LogLevel = (typeof LOG_LEVELS)[number];

interface SystemSnapshot {
    engine_status?: string;
    connection_status?: string;
    incident_state?: string;
    exchange_health_stage?: string;
    log_stream_clients?: number;
    log_buffer_size?: number;
}

interface OperatorAction {
    action: string;
    reason?: string;
    timestamp: string;
    operator?: string;
}

interface OperatorActionsPayload {
    actions: OperatorAction[];
}

interface LogEntry {
    timestamp: string;
    level: LogLevel | string;
    module: string;
    logger: string;
    message: string;
    line?: number;
}

interface LogSnapshotPayload {
    entries: LogEntry[];
    available_modules: string[];
    active_stream_clients: number;
    buffered_entries: number;
}

async function fetcher<T>(url: string): Promise<T> {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }
    const payload = await response.json();
    return unwrapApiData<T>(payload);
}

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlightText(text: string, search: string): ReactNode {
    if (!search.trim()) {
        return text;
    }

    const pattern = new RegExp(`(${escapeRegExp(search)})`, 'ig');
    const parts = text.split(pattern);
    const normalizedSearch = search.toLowerCase();

    return (
        <>
            {parts.map((part, index) => (
                part.toLowerCase() === normalizedSearch ? (
                    <mark key={`${part}-${index}`} className="bg-[#c4a052]/25 text-[#f3d78d] rounded px-0.5">
                        {part}
                    </mark>
                ) : (
                    <span key={`${part}-${index}`}>{part}</span>
                )
            ))}
        </>
    );
}

function playCriticalAlert(): void {
    if (typeof window === 'undefined') {
        return;
    }

    const AudioContextImpl = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextImpl) {
        return;
    }

    const audioContext = new AudioContextImpl();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.type = 'square';
    oscillator.frequency.value = 720;
    gainNode.gain.value = 0.02;

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.08);
    oscillator.onended = () => {
        void audioContext.close();
    };
}

export default function LogsPage() {
    const [entries, setEntries] = useState<LogEntry[]>([]);
    const [availableModules, setAvailableModules] = useState<string[]>([]);
    const [selectedLevels, setSelectedLevels] = useState<LogLevel[]>(['INFO', 'WARNING', 'ERROR', 'CRITICAL']);
    const [selectedModule, setSelectedModule] = useState<string>('ALL');
    const [search, setSearch] = useState('');
    const [streamStatus, setStreamStatus] = useState<'connecting' | 'connected' | 'reconnecting'>('connecting');
    const [autoScroll, setAutoScroll] = useState(true);
    const [pendingCount, setPendingCount] = useState(0);

    const logContainerRef = useRef<HTMLDivElement | null>(null);

    const { data: system, error: systemError } = useSWR<SystemSnapshot>(
        `${API_URL}/api/system`,
        fetcher,
        { refreshInterval: 10000, revalidateOnFocus: false }
    );
    const { data: operatorPayload } = useSWR<OperatorActionsPayload>(
        `${API_URL}/api/operator/actions?limit=10`,
        fetcher,
        { refreshInterval: 10000, revalidateOnFocus: false }
    );

    const recentActions = operatorPayload?.actions ?? [];

    const moduleOptions = useMemo(() => {
        const modules = new Set<string>(availableModules);
        entries.forEach((entry) => {
            if (entry.module) {
                modules.add(entry.module);
            }
        });
        return ['ALL', ...Array.from(modules).sort()];
    }, [availableModules, entries]);

    const filteredEntries = useMemo(() => {
        const levelSet = new Set(selectedLevels);
        return entries.filter((entry) => {
            const levelMatches = levelSet.has(entry.level as LogLevel);
            const moduleMatches = selectedModule === 'ALL' || entry.module === selectedModule;
            const searchMatches = !search.trim() || `${entry.message} ${entry.module} ${entry.logger}`.toLowerCase().includes(search.toLowerCase());
            return levelMatches && moduleMatches && searchMatches;
        });
    }, [entries, search, selectedLevels, selectedModule]);

    const toggleLevel = useCallback((level: LogLevel) => {
        setSelectedLevels((current) => {
            if (current.includes(level)) {
                return current.filter((item) => item !== level);
            }
            return [...current, level];
        });
    }, []);

    const jumpToLatest = useCallback(() => {
        const container = logContainerRef.current;
        if (!container) {
            return;
        }
        container.scrollTop = container.scrollHeight;
        setAutoScroll(true);
        setPendingCount(0);
    }, []);

    const exportLogs = useCallback(() => {
        const body = filteredEntries
            .map((entry) => `[${entry.timestamp}] ${entry.level.padEnd(8)} ${entry.module}: ${entry.message}`)
            .join('\n');
        const blob = new Blob([body], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `cryptoboss-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
        link.click();
        URL.revokeObjectURL(url);
    }, [filteredEntries]);

    useEffect(() => {
        const params = new URLSearchParams({
            limit: '200',
            levels: selectedLevels.join(','),
        });
        if (selectedModule !== 'ALL') {
            params.set('module', selectedModule);
        }

        const stream = new EventSource(`${API_URL}/api/logs/stream?${params.toString()}`);
        setStreamStatus('connecting');

        const handleSnapshot = (event: Event) => {
            const payload = JSON.parse((event as MessageEvent<string>).data) as LogSnapshotPayload;
            setEntries(payload.entries ?? []);
            setAvailableModules(payload.available_modules ?? []);
            setPendingCount(0);
            setStreamStatus('connected');
        };

        const handleLog = (event: Event) => {
            const entry = JSON.parse((event as MessageEvent<string>).data) as LogEntry;
            setEntries((current) => [...current, entry].slice(-400));
            setAvailableModules((current) => Array.from(new Set([...current, entry.module].filter(Boolean))).sort());
            if (entry.level === 'ERROR' || entry.level === 'CRITICAL') {
                playCriticalAlert();
            }
            if (!autoScroll) {
                setPendingCount((count) => count + 1);
            }
            setStreamStatus('connected');
        };

        const handleHeartbeat = () => {
            setStreamStatus('connected');
        };

        stream.addEventListener('snapshot', handleSnapshot);
        stream.addEventListener('log', handleLog);
        stream.addEventListener('heartbeat', handleHeartbeat);
        stream.onerror = () => {
            setStreamStatus('reconnecting');
        };

        return () => {
            stream.removeEventListener('snapshot', handleSnapshot);
            stream.removeEventListener('log', handleLog);
            stream.removeEventListener('heartbeat', handleHeartbeat);
            stream.close();
        };
    }, [autoScroll, selectedLevels, selectedModule]);

    useEffect(() => {
        if (!autoScroll) {
            return;
        }
        const container = logContainerRef.current;
        if (!container) {
            return;
        }
        container.scrollTop = container.scrollHeight;
        setPendingCount(0);
    }, [autoScroll, filteredEntries]);

    const handleScroll = useCallback(() => {
        const container = logContainerRef.current;
        if (!container) {
            return;
        }
        const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 32;
        setAutoScroll(nearBottom);
        if (nearBottom) {
            setPendingCount(0);
        }
    }, []);

    return (
        <div className="space-y-6">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <h1 className="heading-lg mb-1">Live Logs</h1>
                    <p className="text-[#8b98a5] text-sm">
                        Streaming runtime diagnostics with operator context layered in.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <span className={`badge ${streamStatus === 'connected' ? 'badge-success' : streamStatus === 'connecting' ? 'badge-warning' : 'badge-danger'}`}>
                        {streamStatus.toUpperCase()}
                    </span>
                    <span className="badge badge-neutral">
                        {filteredEntries.length} visible
                    </span>
                </div>
            </div>

            {systemError && (
                <div className="card border border-[#a65454] text-[#d28383] text-sm">
                    Failed to load runtime summary: {systemError.message}
                </div>
            )}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Engine</span>
                    </div>
                    <span className={`badge ${system?.engine_status === 'running' ? 'badge-success' : system?.engine_status === 'paused' ? 'badge-warning' : 'badge-danger'}`}>
                        {(system?.engine_status || 'stopped').toUpperCase()}
                    </span>
                </div>
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Connection</span>
                    </div>
                    <span className={`badge ${system?.connection_status === 'connected' ? 'badge-success' : system?.connection_status === 'connecting' ? 'badge-warning' : 'badge-danger'}`}>
                        {(system?.connection_status || 'disconnected').toUpperCase()}
                    </span>
                </div>
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Exchange Stage</span>
                    </div>
                    <span className="badge badge-neutral">
                        {(system?.exchange_health_stage || 'UNKNOWN').replace('_', ' ')}
                    </span>
                </div>
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Stream Clients</span>
                    </div>
                    <div className="text-2xl font-semibold text-[#e7e9ea]">
                        {system?.log_stream_clients ?? 0}
                    </div>
                    <div className="text-xs text-[#6b7280] mt-1">
                        Buffer: {system?.log_buffer_size ?? 0}
                    </div>
                </div>
            </div>

            <div className="card space-y-4">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                    <div className="flex flex-wrap gap-2">
                        {LOG_LEVELS.map((level) => {
                            const active = selectedLevels.includes(level);
                            const levelTone =
                                level === 'CRITICAL' || level === 'ERROR'
                                    ? active ? 'bg-[#a65454] text-white border-[#a65454]' : 'text-[#d28383] border-[#5a2a2a]'
                                    : level === 'WARNING'
                                        ? active ? 'bg-[#c4a052] text-[#0f1318] border-[#c4a052]' : 'text-[#e1c981] border-[#5b4a1d]'
                                        : active ? 'bg-[#4a9268] text-white border-[#4a9268]' : 'text-[#8b98a5] border-[#2d3640]';
                            return (
                                <button
                                    key={level}
                                    type="button"
                                    onClick={() => toggleLevel(level)}
                                    className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${levelTone}`}
                                >
                                    {level}
                                </button>
                            );
                        })}
                    </div>

                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                        <select
                            value={selectedModule}
                            onChange={(event) => setSelectedModule(event.target.value)}
                            className="rounded-md border border-[#2d3640] bg-[#131921] px-3 py-2 text-sm text-[#e7e9ea] outline-none"
                        >
                            {moduleOptions.map((module) => (
                                <option key={module} value={module}>
                                    {module === 'ALL' ? 'All Modules' : module}
                                </option>
                            ))}
                        </select>

                        <input
                            type="text"
                            value={search}
                            onChange={(event) => setSearch(event.target.value)}
                            placeholder="Search message or module"
                            className="w-full rounded-md border border-[#2d3640] bg-[#131921] px-3 py-2 text-sm text-[#e7e9ea] outline-none sm:w-72"
                        />

                        <button
                            type="button"
                            onClick={() => setAutoScroll((value) => !value)}
                            className={`rounded-md border px-3 py-2 text-sm ${autoScroll ? 'border-[#4a9268] text-[#4a9268]' : 'border-[#2d3640] text-[#8b98a5]'}`}
                        >
                            Auto-scroll {autoScroll ? 'On' : 'Off'}
                        </button>

                        <button
                            type="button"
                            onClick={jumpToLatest}
                            className="rounded-md border border-[#2d3640] px-3 py-2 text-sm text-[#e7e9ea]"
                        >
                            Jump to Latest{pendingCount > 0 ? ` (${pendingCount})` : ''}
                        </button>

                        <button
                            type="button"
                            onClick={exportLogs}
                            className="rounded-md border border-[#2d3640] px-3 py-2 text-sm text-[#e7e9ea]"
                        >
                            Export Logs
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                    <div
                        ref={logContainerRef}
                        onScroll={handleScroll}
                        className="max-h-[36rem] overflow-auto rounded-md border border-[#2d3640] bg-[#11171d] font-mono text-sm"
                    >
                        {filteredEntries.length === 0 ? (
                            <div className="p-6 text-[#6b7280]">
                                No log entries match the current filters.
                            </div>
                        ) : (
                            filteredEntries.map((entry, index) => {
                                const isCritical = entry.level === 'CRITICAL' || entry.level === 'ERROR';
                                const rowTone = isCritical
                                    ? 'border-l-[#a65454] bg-[#2a1416]/60'
                                    : entry.level === 'WARNING'
                                        ? 'border-l-[#c4a052] bg-[#211d14]/40'
                                        : 'border-l-[#2d3640]';
                                return (
                                    <div
                                        key={`${entry.timestamp}-${index}`}
                                        className={`border-l-2 px-4 py-3 ${rowTone} ${entry.level === 'CRITICAL' ? 'animate-pulse' : ''}`}
                                    >
                                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                                            <div className="flex flex-wrap items-center gap-2 text-xs">
                                                <span className="text-[#8b98a5]">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                                                <span className={`badge ${isCritical ? 'badge-danger' : entry.level === 'WARNING' ? 'badge-warning' : 'badge-neutral'}`}>
                                                    {entry.level}
                                                </span>
                                                <span className="text-[#c9d1d9]">{entry.module}</span>
                                                <span className="text-[#6b7280]">{entry.logger}</span>
                                                {entry.line ? <span className="text-[#6b7280]">L{entry.line}</span> : null}
                                            </div>
                                        </div>
                                        <div className="mt-2 whitespace-pre-wrap break-words text-[#e7e9ea]">
                                            {highlightText(entry.message, search)}
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>

                    <div className="space-y-4">
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Recent Operator Actions</span>
                                <span className="badge badge-neutral">{recentActions.length}</span>
                            </div>
                            <div className="space-y-3">
                                {recentActions.length === 0 ? (
                                    <div className="text-sm text-[#6b7280]">No operator actions recorded.</div>
                                ) : (
                                    recentActions.slice().reverse().map((action, index) => (
                                        <div key={`${action.timestamp}-${index}`} className="rounded-md border border-[#2d3640] p-3">
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="text-sm font-medium text-[#e7e9ea]">{action.action}</span>
                                                <span className="text-xs text-[#6b7280]">
                                                    {new Date(action.timestamp).toLocaleString()}
                                                </span>
                                            </div>
                                            <div className="mt-2 text-sm text-[#8b98a5]">
                                                {action.reason || 'No reason captured'}
                                            </div>
                                            {action.operator ? (
                                                <div className="mt-2 text-xs text-[#6b7280]">
                                                    Operator: {action.operator}
                                                </div>
                                            ) : null}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Search Context</span>
                            </div>
                            <div className="space-y-2 text-sm text-[#8b98a5]">
                                <div>Module: <span className="text-[#e7e9ea]">{selectedModule}</span></div>
                                <div>Levels: <span className="text-[#e7e9ea]">{selectedLevels.join(', ') || 'None'}</span></div>
                                <div>Incident: <span className="text-[#e7e9ea]">{system?.incident_state || 'NORMAL'}</span></div>
                                <div>Search: <span className="text-[#e7e9ea]">{search.trim() || 'None'}</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
