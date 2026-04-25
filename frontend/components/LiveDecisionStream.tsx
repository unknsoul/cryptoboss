"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { unwrapApiData } from "@/lib/api";

interface TradeDecision {
    decision_id: string;
    intent_id: string;
    timestamp: string;
    symbol: string;
    direction: string;
    status: "approved" | "rejected";
    rejection_stage?: string;
    rejection_reason?: string;
    confidence_score: number;
    market_regime?: string;
    directional_bias?: string;
    executed?: boolean;
    fill_price?: number;
    slippage_bps?: number;
    total_pipeline_ms?: number;
}

interface LiveDecisionStreamProps {
    className?: string;
    maxItems?: number;
    showFilters?: boolean;
}

export default function LiveDecisionStream({
    className = "",
    maxItems = 50,
    showFilters = true,
}: LiveDecisionStreamProps) {
    const [decisions, setDecisions] = useState<TradeDecision[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const [filter, setFilter] = useState<"all" | "approved" | "rejected">("all");
    const [symbolFilter, setSymbolFilter] = useState<string>("");
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connectWebSocket = useCallback(() => {
        const wsUrl =
            process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/v11/stream";

        try {
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                console.log("WebSocket connected");
                setIsConnected(true);
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === "decision") {
                        setDecisions((prev) => {
                            const updated = [data.decision, ...prev].slice(0, maxItems);
                            return updated;
                        });
                    }
                } catch (e) {
                    console.error("Failed to parse message:", e);
                }
            };

            wsRef.current.onclose = () => {
                console.log("WebSocket disconnected");
                setIsConnected(false);

                // Reconnect after 5 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    connectWebSocket();
                }, 5000);
            };

            wsRef.current.onerror = (error) => {
                console.error("WebSocket error:", error);
            };
        } catch (e) {
            console.error("Failed to connect:", e);
        }
    }, [maxItems]);

    useEffect(() => {
        connectWebSocket();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
        };
    }, [connectWebSocket]);

    // Load initial decisions
    useEffect(() => {
        async function loadDecisions() {
            try {
                const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
                const res = await fetch(`${API_URL}/api/v11/decisions?limit=20`);
                if (res.ok) {
                    const payload = await res.json();
                    const data: any = unwrapApiData(payload);
                    setDecisions(data.decisions || []);
                }
            } catch (e) {
                console.error("Failed to load decisions:", e);
            }
        }
        loadDecisions();
    }, []);

    const filteredDecisions = decisions.filter((d) => {
        if (filter !== "all" && d.status !== filter) return false;
        if (symbolFilter && !d.symbol.includes(symbolFilter.toUpperCase()))
            return false;
        return true;
    });

    const getStatusColor = (decision: TradeDecision) => {
        if (decision.status === "rejected") return "text-red-400";
        if (decision.executed) return "text-green-400";
        return "text-yellow-400";
    };

    const getStatusIcon = (decision: TradeDecision) => {
        if (decision.status === "rejected") return "✗";
        if (decision.executed) return "✓";
        return "⏳";
    };

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString();
    };

    return (
        <div className={`bg-gray-900/50 rounded-xl border border-gray-800 ${className}`}>
            {/* Header */}
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <h2 className="text-lg font-semibold text-white">
                        Live Decision Stream
                    </h2>
                    <span
                        className={`px-2 py-1 rounded-full text-xs ${isConnected
                            ? "bg-green-500/20 text-green-400"
                            : "bg-red-500/20 text-red-400"
                            }`}
                    >
                        {isConnected ? "● Connected" : "○ Disconnected"}
                    </span>
                </div>

                {showFilters && (
                    <div className="flex gap-2">
                        <select
                            value={filter}
                            onChange={(e) => setFilter(e.target.value as typeof filter)}
                            className="bg-gray-800 text-gray-300 text-sm rounded px-2 py-1 border border-gray-700"
                        >
                            <option value="all">All</option>
                            <option value="approved">Approved</option>
                            <option value="rejected">Rejected</option>
                        </select>

                        <input
                            type="text"
                            placeholder="Symbol..."
                            value={symbolFilter}
                            onChange={(e) => setSymbolFilter(e.target.value)}
                            className="bg-gray-800 text-gray-300 text-sm rounded px-2 py-1 border border-gray-700 w-24"
                        />
                    </div>
                )}
            </div>

            {/* Decision List */}
            <div className="max-h-[600px] overflow-y-auto">
                {filteredDecisions.length === 0 ? (
                    <div className="p-8 text-center text-gray-500">
                        <div className="text-4xl mb-2">📊</div>
                        <p>No decisions yet. Waiting for trade signals...</p>
                    </div>
                ) : (
                    <div className="divide-y divide-gray-800">
                        {filteredDecisions.map((decision) => (
                            <div
                                key={decision.decision_id}
                                className="p-4 hover:bg-gray-800/50 transition-colors"
                            >
                                <div className="flex items-start justify-between">
                                    {/* Left: Main info */}
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className={`text-lg ${getStatusColor(decision)}`}>
                                                {getStatusIcon(decision)}
                                            </span>
                                            <span className="font-mono font-bold text-white">
                                                {decision.symbol}
                                            </span>
                                            <span
                                                className={`text-sm font-medium ${decision.direction === "long"
                                                    ? "text-green-400"
                                                    : "text-red-400"
                                                    }`}
                                            >
                                                {decision.direction.toUpperCase()}
                                            </span>
                                            <span className="text-gray-500 text-xs">
                                                {formatTime(decision.timestamp)}
                                            </span>
                                        </div>

                                        {decision.status === "rejected" && decision.rejection_reason && (
                                            <div className="text-red-400/80 text-sm mt-1">
                                                ⛔ {decision.rejection_stage}: {decision.rejection_reason}
                                            </div>
                                        )}

                                        {decision.executed && decision.fill_price && (
                                            <div className="text-green-400/80 text-sm mt-1">
                                                Filled @ ${decision.fill_price.toFixed(2)}
                                                {decision.slippage_bps !== undefined && (
                                                    <span className="text-gray-500 ml-2">
                                                        ({decision.slippage_bps.toFixed(1)} bps slippage)
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    {/* Right: Metrics */}
                                    <div className="text-right">
                                        <div className="text-sm">
                                            <span className="text-gray-500">Confidence: </span>
                                            <span
                                                className={`font-mono ${decision.confidence_score >= 0.7
                                                    ? "text-green-400"
                                                    : decision.confidence_score >= 0.5
                                                        ? "text-yellow-400"
                                                        : "text-red-400"
                                                    }`}
                                            >
                                                {(decision.confidence_score * 100).toFixed(0)}%
                                            </span>
                                        </div>

                                        {decision.market_regime && (
                                            <div className="text-xs text-gray-500 mt-1">
                                                {decision.market_regime} • {decision.directional_bias}
                                            </div>
                                        )}

                                        {decision.total_pipeline_ms !== undefined && (
                                            <div className="text-xs text-gray-600 mt-1">
                                                {decision.total_pipeline_ms.toFixed(0)}ms
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Footer Stats */}
            <div className="p-3 border-t border-gray-800 flex justify-between text-xs text-gray-500">
                <span>
                    Showing {filteredDecisions.length} of {decisions.length} decisions
                </span>
                <span>
                    {decisions.filter((d) => d.status === "approved").length} approved •{" "}
                    {decisions.filter((d) => d.executed).length} executed
                </span>
            </div>
        </div>
    );
}
