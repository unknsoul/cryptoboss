'use client';

import React from 'react';

/**
 * DecisionNarrative - v11.1 FINAL-MAP
 * 
 * Displays the latest trading decision with full explanation.
 * Shows why trade was allowed, blocked, or size reduced.
 */

interface DecisionNarrativeProps {
    decisionResult: 'NO_TRADE' | 'TRADE_ALLOWED' | 'SIZE_REDUCED' | null;
    failedGate: string | null;
    reason: string;
    metrics?: Record<string, number>;
    symbol?: string;
    timestamp?: string;
}

const DecisionNarrative: React.FC<DecisionNarrativeProps> = ({
    decisionResult,
    failedGate,
    reason,
    metrics = {},
    symbol,
    timestamp,
}) => {
    const config = {
        NO_TRADE: {
            icon: '🚫',
            color: '#ff4444',
            bg: 'rgba(255, 68, 68, 0.1)',
            label: 'Trade Blocked',
        },
        TRADE_ALLOWED: {
            icon: '✅',
            color: '#00ff64',
            bg: 'rgba(0, 255, 100, 0.1)',
            label: 'Trade Approved',
        },
        SIZE_REDUCED: {
            icon: '⚠️',
            color: '#ffaa00',
            bg: 'rgba(255, 170, 0, 0.1)',
            label: 'Size Reduced',
        },
    }[decisionResult || 'NO_TRADE'] || {
        icon: '❓',
        color: '#888',
        bg: 'rgba(128, 128, 128, 0.1)',
        label: 'Unknown',
    };

    return (
        <div className="decision-narrative" style={{ borderColor: config.color }}>
            <div className="decision-narrative__header">
                <span className="decision-narrative__icon">{config.icon}</span>
                <div className="decision-narrative__title">
                    <h3 style={{ color: config.color }}>{config.label}</h3>
                    {symbol && <span className="decision-narrative__symbol">{symbol}</span>}
                </div>
                {timestamp && (
                    <span className="decision-narrative__time">
                        {new Date(timestamp).toLocaleTimeString()}
                    </span>
                )}
            </div>

            <div className="decision-narrative__body" style={{ background: config.bg }}>
                {failedGate && (
                    <div className="decision-narrative__gate">
                        <span className="decision-narrative__gate-label">Failed Gate:</span>
                        <span className="decision-narrative__gate-name" style={{ color: config.color }}>
                            {failedGate}
                        </span>
                    </div>
                )}

                <p className="decision-narrative__reason">{reason}</p>

                {Object.keys(metrics).length > 0 && (
                    <div className="decision-narrative__metrics">
                        {Object.entries(metrics).map(([key, value]) => (
                            <div key={key} className="decision-narrative__metric">
                                <span className="decision-narrative__metric-key">
                                    {key.replace(/_/g, ' ')}:
                                </span>
                                <span className="decision-narrative__metric-value">
                                    {typeof value === 'number' ? value.toFixed(4) : value}
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <style jsx>{`
        .decision-narrative {
          background: var(--bg-secondary, #1a1a2e);
          border-radius: 12px;
          padding: 16px;
          border-left: 4px solid;
        }

        .decision-narrative__header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 12px;
        }

        .decision-narrative__icon {
          font-size: 28px;
        }

        .decision-narrative__title h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }

        .decision-narrative__symbol {
          background: var(--bg-tertiary, #252542);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 11px;
          margin-left: 8px;
        }

        .decision-narrative__time {
          margin-left: auto;
          color: var(--text-secondary, #888);
          font-size: 12px;
        }

        .decision-narrative__body {
          padding: 12px;
          border-radius: 8px;
        }

        .decision-narrative__gate {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
          font-size: 13px;
        }

        .decision-narrative__gate-label {
          color: var(--text-secondary, #888);
        }

        .decision-narrative__gate-name {
          font-weight: 600;
          font-family: monospace;
        }

        .decision-narrative__reason {
          margin: 0;
          color: var(--text-primary, #fff);
          font-size: 14px;
          line-height: 1.5;
        }

        .decision-narrative__metrics {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid var(--border-color, #333);
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 8px;
        }

        .decision-narrative__metric {
          display: flex;
          flex-direction: column;
          font-size: 12px;
        }

        .decision-narrative__metric-key {
          color: var(--text-secondary, #888);
          text-transform: capitalize;
        }

        .decision-narrative__metric-value {
          color: var(--text-primary, #fff);
          font-family: monospace;
        }
      `}</style>
        </div>
    );
};

export default DecisionNarrative;
