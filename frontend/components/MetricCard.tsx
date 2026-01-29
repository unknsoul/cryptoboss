'use client';

import React from 'react';

/**
 * MetricCard - v11.1 FINAL-MAP
 * 
 * Displays a metric with data trust visualization.
 * Visual indicators for data source and staleness.
 */

type DataTag = 'LIVE_EXCHANGE' | 'TESTNET_EXCHANGE' | 'DERIVED' | 'SIMULATED' | 'STALE';

interface MetricCardProps {
    label: string;
    value: string | number;
    change?: number;
    changeLabel?: string;
    icon?: string;
    dataTag?: DataTag;
    timestamp?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
    label,
    value,
    change,
    changeLabel,
    icon,
    dataTag = 'LIVE_EXCHANGE',
    timestamp,
}) => {
    const tagConfig: Record<DataTag, { label: string; color: string; style: string }> = {
        LIVE_EXCHANGE: { label: 'LIVE', color: '#00ff64', style: 'solid' },
        TESTNET_EXCHANGE: { label: 'TEST', color: '#888', style: 'solid' },
        DERIVED: { label: 'DERIVED', color: '#888', style: 'solid' },
        SIMULATED: { label: 'SIM', color: '#ffaa00', style: 'dashed' },
        STALE: { label: 'STALE', color: '#ff4444', style: 'dashed' },
    };

    const config = tagConfig[dataTag];
    const isStale = dataTag === 'STALE';
    const isSimulated = dataTag === 'SIMULATED';

    const formatValue = (val: string | number) => {
        if (typeof val === 'number') {
            return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
        return val;
    };

    return (
        <div
            className={`metric-card ${isStale ? 'metric-card--stale' : ''} ${isSimulated ? 'metric-card--simulated' : ''
                }`}
            style={{ borderStyle: config.style }}
        >
            {isStale && (
                <div className="metric-card__stale-warning">
                    ⚠️ STALE DATA - May not reflect current state
                </div>
            )}

            <div className="metric-card__header">
                {icon && <span className="metric-card__icon">{icon}</span>}
                <span className="metric-card__label">{label}</span>
                <span
                    className="metric-card__tag"
                    style={{ color: config.color, borderColor: config.color }}
                >
                    {config.label}
                </span>
            </div>

            <div className="metric-card__value">{formatValue(value)}</div>

            {change !== undefined && (
                <div
                    className="metric-card__change"
                    style={{ color: change >= 0 ? '#00ff64' : '#ff4444' }}
                >
                    {change >= 0 ? '▲' : '▼'} {Math.abs(change).toFixed(2)}%
                    {changeLabel && <span className="metric-card__change-label">{changeLabel}</span>}
                </div>
            )}

            {timestamp && (
                <div className="metric-card__timestamp">
                    Updated: {new Date(timestamp).toLocaleTimeString()}
                </div>
            )}

            <style jsx>{`
        .metric-card {
          background: var(--bg-secondary, #1a1a2e);
          border: 1px solid var(--border-color, #333);
          border-radius: 12px;
          padding: 16px;
          position: relative;
          overflow: hidden;
        }

        .metric-card--stale {
          border-color: #ff4444 !important;
          background: rgba(255, 68, 68, 0.05);
        }

        .metric-card--simulated {
          border-style: dashed !important;
          border-color: #ffaa00 !important;
        }

        .metric-card__stale-warning {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          background: #ff4444;
          color: #000;
          font-size: 10px;
          font-weight: 700;
          text-align: center;
          padding: 2px;
        }

        .metric-card__header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .metric-card__icon {
          font-size: 16px;
        }

        .metric-card__label {
          font-size: 12px;
          color: var(--text-secondary, #888);
          flex: 1;
        }

        .metric-card__tag {
          font-size: 9px;
          font-weight: 700;
          padding: 2px 6px;
          border: 1px solid;
          border-radius: 4px;
        }

        .metric-card__value {
          font-size: 24px;
          font-weight: 700;
          color: var(--text-primary, #fff);
          margin-bottom: 4px;
        }

        .metric-card__change {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 13px;
          font-weight: 500;
        }

        .metric-card__change-label {
          color: var(--text-secondary, #888);
          font-size: 11px;
          margin-left: 4px;
        }

        .metric-card__timestamp {
          font-size: 10px;
          color: var(--text-tertiary, #666);
          margin-top: 8px;
        }
      `}</style>
        </div>
    );
};

export default MetricCard;
