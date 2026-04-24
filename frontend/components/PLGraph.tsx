'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface PLPoint {
    time: string;
    pnl: number;
    trade_pnl: number;
    symbol: string;
}

interface PLSummary {
    points: PLPoint[];
    total_pnl: number;
    total_trades: number;
    win_rate: number;
    best_trade: number;
    worst_trade: number;
}

const EMPTY: PLSummary = {
    points: [],
    total_pnl: 0,
    total_trades: 0,
    win_rate: 0,
    best_trade: 0,
    worst_trade: 0,
};

export function PLGraph({ apiUrl }: { apiUrl: string }) {
    const [data, setData] = useState<PLSummary>(EMPTY);
    const [loading, setLoading] = useState(true);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);

    const fetchPnl = useCallback(async () => {
        try {
            const r = await fetch(`${apiUrl}/api/pnl/history`);
            if (r.ok) {
                const payload = await r.json();
                // Handle wrapped response (data field) or direct response
                const d = payload?.data ?? payload;
                setData({
                    points: Array.isArray(d?.points) ? d.points : [],
                    total_pnl: typeof d?.total_pnl === 'number' ? d.total_pnl : 0,
                    total_trades: typeof d?.total_trades === 'number' ? d.total_trades : 0,
                    win_rate: typeof d?.win_rate === 'number' ? d.win_rate : 0,
                    best_trade: typeof d?.best_trade === 'number' ? d.best_trade : 0,
                    worst_trade: typeof d?.worst_trade === 'number' ? d.worst_trade : 0,
                });
            }
        } catch {
            /* Graph intentionally fails open and shows empty-state text. */
        } finally {
            setLoading(false);
        }
    }, [apiUrl]);

    useEffect(() => {
        fetchPnl();
        const iv = setInterval(fetchPnl, 10_000);
        return () => clearInterval(iv);
    }, [fetchPnl]);

    // Canvas rendering
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const pts = data.points;
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (pts.length < 2) {
            ctx.fillStyle = '#374151';
            ctx.font = '12px system-ui';
            ctx.textAlign = 'center';
            ctx.fillText(
                'No closed trades yet — graph populates as trades complete',
                W / 2,
                H / 2
            );
            return;
        }

        const vals = pts.map((p) => p.pnl);
        const minV = Math.min(...vals, 0);
        const maxV = Math.max(...vals, 0);
        const range = maxV - minV || 1;
        const PX = 8;
        const PY = 10;
        const toY = (v: number) =>
            PY + (H - PY * 2) - ((v - minV) / range) * (H - PY * 2);
        const toX = (i: number) =>
            PX + (i / (pts.length - 1)) * (W - PX * 2);
        const isPos = data.total_pnl >= 0;
        const lineColor = isPos ? '#4a9268' : '#e06c75';

        // Grid lines
        ctx.strokeStyle = '#1e2530';
        ctx.lineWidth = 1;
        [0.25, 0.5, 0.75].forEach((t) => {
            const y = PY + t * (H - PY * 2);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        });

        // Zero line
        ctx.strokeStyle = '#3d4d5c';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, toY(0));
        ctx.lineTo(W, toY(0));
        ctx.stroke();
        ctx.setLineDash([]);

        // Gradient fill
        const grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(
            0,
            isPos ? 'rgba(74,146,104,0.25)' : 'rgba(224,108,117,0.25)'
        );
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(0));
        pts.forEach((p, i) => ctx.lineTo(toX(i), toY(p.pnl)));
        ctx.lineTo(toX(pts.length - 1), toY(0));
        ctx.closePath();
        ctx.fill();

        // Line
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        pts.forEach((p, i) =>
            i === 0
                ? ctx.moveTo(toX(i), toY(p.pnl))
                : ctx.lineTo(toX(i), toY(p.pnl))
        );
        ctx.stroke();

        // Terminal dot
        const lx = toX(pts.length - 1);
        const ly = toY(pts[pts.length - 1].pnl);
        ctx.fillStyle = lineColor;
        ctx.beginPath();
        ctx.arc(lx, ly, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0f1419';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }, [data]);

    // Tooltip on hover
    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        const tooltip = tooltipRef.current;
        if (!canvas || !tooltip || data.points.length < 2) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const idx = Math.max(
            0,
            Math.min(
                data.points.length - 1,
                Math.round(
                    ((x - 8) / (canvas.width - 16)) * (data.points.length - 1)
                )
            )
        );
        const pt = data.points[idx];
        const sign = pt.trade_pnl >= 0 ? '+' : '';
        tooltip.style.display = 'block';
        tooltip.style.left = `${Math.min(x + 12, rect.width - 140)}px`;
        tooltip.style.top = `${e.clientY - rect.top - 32}px`;
        tooltip.innerHTML = `
            <div style="font-size:11px;color:#8b98a5;margin-bottom:2px">${pt.symbol || '—'}</div>
            <div style="font-size:12px;font-weight:600;color:${pt.pnl >= 0 ? '#4a9268' : '#e06c75'}">
                Total: ${pt.pnl >= 0 ? '+' : ''}${pt.pnl.toFixed(2)} USDT</div>
            <div style="font-size:11px;color:${pt.trade_pnl >= 0 ? '#4a9268' : '#e06c75'}">
                Trade: ${sign}${pt.trade_pnl.toFixed(2)} USDT</div>`;
    };

    const pnlColor =
        data.total_pnl >= 0 ? 'text-[#4a9268]' : 'text-[#e06c75]';

    return (
        <div className="card">
            <div className="card-header flex items-center justify-between flex-wrap gap-3 mb-3">
                <span className="card-title">Cumulative P/L</span>
                <div className="flex items-center gap-5 text-sm flex-wrap">
                    <span
                        className={`font-mono font-semibold text-base ${pnlColor}`}
                    >
                        {data.total_pnl >= 0 ? '+' : ''}
                        {data.total_pnl.toFixed(2)} USDT
                    </span>
                    <span className="text-[#8b98a5]">
                        Trades:{' '}
                        <span className="text-[#e7e9ea]">
                            {data.total_trades}
                        </span>
                    </span>
                    <span className="text-[#8b98a5]">
                        Win Rate:{' '}
                        <span
                            className={
                                data.win_rate >= 55
                                    ? 'text-[#4a9268]'
                                    : 'text-[#c4a052]'
                            }
                        >
                            {data.win_rate.toFixed(1)}%
                        </span>
                    </span>
                </div>
            </div>
            <div className="relative" style={{ height: 112 }}>
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-[#6b7280] text-sm animate-pulse">
                            Loading P/L history...
                        </div>
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                    width={900}
                    height={112}
                    className="w-full h-full cursor-crosshair"
                    style={{ display: 'block' }}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={() => {
                        if (tooltipRef.current)
                            tooltipRef.current.style.display = 'none';
                    }}
                />
                <div
                    ref={tooltipRef}
                    className="pointer-events-none absolute z-10 hidden rounded-md border border-[#2d3640] bg-[#1a1f26] px-2.5 py-1.5 shadow-lg"
                    style={{ minWidth: 135 }}
                />
            </div>
        </div>
    );
}
