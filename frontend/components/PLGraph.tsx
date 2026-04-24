'use client';

import { useEffect, useRef, useState } from 'react';

interface PLPoint {
    time: string;
    pnl: number;
    trade_pnl: number;
    symbol: string;
}

export function PLGraph({ apiUrl }: { apiUrl: string }) {
    const [points, setPoints] = useState<PLPoint[]>([]);
    const [totalPnl, setTotalPnl] = useState(0);
    const [winRate, setWinRate] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const fetchPnl = async () => {
            try {
                const response = await fetch(`${apiUrl}/api/pnl/history`);
                const payload = await response.json();
                const data = payload?.data ?? payload;

                setPoints(Array.isArray(data?.points) ? data.points : []);
                setTotalPnl(typeof data?.total_pnl === 'number' ? data.total_pnl : 0);
                setWinRate(typeof data?.win_rate === 'number' ? data.win_rate : 0);
            } catch {
                // Graph intentionally fails open and shows empty-state text.
            }
        };

        fetchPnl();
        const interval = setInterval(fetchPnl, 10000);
        return () => clearInterval(interval);
    }, [apiUrl]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || points.length < 2) {
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        const width = canvas.width;
        const height = canvas.height;
        const values = points.map((p) => p.pnl);
        const minV = Math.min(...values, 0);
        const maxV = Math.max(...values, 0);
        const range = maxV - minV || 1;
        const toY = (v: number) => height - 10 - ((v - minV) / range) * (height - 20);
        const toX = (index: number) => (index / (points.length - 1)) * width;

        ctx.clearRect(0, 0, width, height);

        const zeroY = toY(0);
        ctx.strokeStyle = '#2d3640';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, zeroY);
        ctx.lineTo(width, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);

        const isPositive = totalPnl >= 0;
        ctx.strokeStyle = isPositive ? '#4a9268' : '#e06c75';
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(toX(index), toY(point.pnl));
                return;
            }
            ctx.lineTo(toX(index), toY(point.pnl));
        });
        ctx.stroke();

        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, isPositive ? 'rgba(74,146,104,0.3)' : 'rgba(224,108,117,0.3)');
        gradient.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = gradient;

        ctx.beginPath();
        points.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(toX(index), toY(point.pnl));
                return;
            }
            ctx.lineTo(toX(index), toY(point.pnl));
        });
        ctx.lineTo(width, zeroY);
        ctx.lineTo(0, zeroY);
        ctx.closePath();
        ctx.fill();
    }, [points, totalPnl]);

    return (
        <div className="card">
            <div className="card-header flex items-center justify-between">
                <span className="card-title">Cumulative P/L</span>
                <div className="flex gap-4 text-sm">
                    <span className={totalPnl >= 0 ? 'text-[#4a9268]' : 'text-[#e06c75]'}>
                        {totalPnl >= 0 ? '+' : ''}
                        {totalPnl.toFixed(2)} USDT
                    </span>
                    <span className="text-[#8b98a5]">
                        Win Rate: <span className="text-[#e7e9ea]">{winRate.toFixed(1)}%</span>
                    </span>
                </div>
            </div>

            {points.length < 2 ? (
                <div className="flex items-center justify-center h-24 text-[#6b7280] text-sm">
                    No trade history yet - P/L graph will populate as trades close
                </div>
            ) : (
                <canvas ref={canvasRef} width={600} height={100} className="w-full h-24 mt-2" />
            )}
        </div>
    );
}
