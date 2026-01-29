'use client';

import React, { useMemo } from 'react';

/**
 * WhyNoTrade Panel - v10.3 Explainability
 * 
 * Shows a visual breakdown of which gate blocked a trade,
 * making it clear why the system did not execute.
 */

interface GateResult {
    gate: string;
    passed: boolean;
    reason: string;
    details?: Record<string, unknown>;
}

interface WhyNoTradeProps {
    gateResults: GateResult[];
    symbol?: string;
    timestamp?: string;
    direction?: string;
}

const gateLabels: Record<string, { label: string; icon: string; description: string }> = {
    'INCIDENT_GATE': {
        label: 'Incident Check',
        icon: '🚨',
        description: 'System incident state (normal/degraded/freeze/halt)',
    },
    'HOURS_GATE': {
        label: 'Trading Hours',
        icon: '🕐',
        description: 'Market hours and session restrictions',
    },
    'RISK_CHECK': {
        label: 'Risk Assessment',
        icon: '⚠️',
        description: 'Risk limits, exposure, and position sizing',
    },
    'MARKET_ANALYSIS': {
        label: 'Market Context',
        icon: '📊',
        description: 'Market regime and volatility analysis',
    },
    'BIAS_ALIGNMENT': {
        label: 'Bias Alignment',
        icon: '🧭',
        description: 'Directional bias confirmation',
    },
    'CAPITAL_PERMISSION': {
        label: 'Capital Check',
        icon: '💰',
        description: 'Available capital and allocation limits',
    },
    'ML_CONTAINMENT': {
        label: 'ML Validation',
        icon: '🤖',
        description: 'ML model confidence and bounds checking',
    },
    'SPREAD_CHECK': {
        label: 'Spread Check',
        icon: '📈',
        description: 'Bid-ask spread within acceptable limits',
    },
    'SIZE_CALCULATION': {
        label: 'Size Calculation',
        icon: '📐',
        description: 'Position size calculation and validation',
    },
    'FINAL_DECISION': {
        label: 'Final Decision',
        icon: '✅',
        description: 'Final trade decision outcome',
    },
};

const WhyNoTrade: React.FC<WhyNoTradeProps> = ({
    gateResults,
    symbol,
    timestamp,
    direction,
}) => {
    const blockingGate = useMemo(() => {
        return gateResults.find(g => !g.passed);
    }, [gateResults]);

    const passedGates = useMemo(() => {
        return gateResults.filter(g => g.passed);
    }, [gateResults]);

    if (!blockingGate) {
        return (
            <div className="why-no-trade why-no-trade--approved">
                <div className="why-no-trade__header">
                    <span className="why-no-trade__icon">✅</span>
                    <h3>Trade Approved</h3>
                </div>
                <p>All {gateResults.length} gates passed. Trade was executed.</p>
            </div>
        );
    }

    return (
        <div className="why-no-trade why-no-trade--blocked">
            <div className="why-no-trade__header">
                <span className="why-no-trade__icon">🚫</span>
                <h3>Trade Not Executed</h3>
                {symbol && <span className="why-no-trade__symbol">{symbol}</span>}
                {direction && <span className={`why-no-trade__direction why-no-trade__direction--${direction}`}>{direction.toUpperCase()}</span>}
            </div>

            {timestamp && (
                <p className="why-no-trade__timestamp">
                    {new Date(timestamp).toLocaleString()}
                </p>
            )}

            {/* Blocking Gate Highlight */}
            <div className="why-no-trade__blocker">
                <div className="why-no-trade__blocker-header">
                    <span className="why-no-trade__blocker-icon">
                        {gateLabels[blockingGate.gate]?.icon || '❌'}
                    </span>
                    <div>
                        <h4>{gateLabels[blockingGate.gate]?.label || blockingGate.gate}</h4>
                        <p className="why-no-trade__blocker-desc">
                            {gateLabels[blockingGate.gate]?.description || 'Unknown gate'}
                        </p>
                    </div>
                </div>
                <div className="why-no-trade__blocker-reason">
                    <strong>Reason:</strong> {blockingGate.reason}
                </div>
                {blockingGate.details && (
                    <div className="why-no-trade__blocker-details">
                        <pre>{JSON.stringify(blockingGate.details, null, 2)}</pre>
                    </div>
                )}
            </div>

            {/* Gate Pipeline */}
            <div className="why-no-trade__pipeline">
                <h4>Gate Pipeline Status</h4>
                <div className="why-no-trade__gates">
                    {gateResults.map((gate, idx) => (
                        <div
                            key={gate.gate}
                            className={`why-no-trade__gate ${gate.passed
                                    ? 'why-no-trade__gate--passed'
                                    : 'why-no-trade__gate--blocked'
                                }`}
                        >
                            <span className="why-no-trade__gate-number">{idx + 1}</span>
                            <span className="why-no-trade__gate-icon">
                                {gate.passed ? '✓' : '✗'}
                            </span>
                            <span className="why-no-trade__gate-name">
                                {gateLabels[gate.gate]?.label || gate.gate}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Passed Gates Summary */}
            {passedGates.length > 0 && (
                <div className="why-no-trade__passed">
                    <h4>Passed Gates ({passedGates.length})</h4>
                    <div className="why-no-trade__passed-list">
                        {passedGates.map(gate => (
                            <span key={gate.gate} className="why-no-trade__passed-badge">
                                {gateLabels[gate.gate]?.icon || '✓'} {gateLabels[gate.gate]?.label || gate.gate}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            <style jsx>{`
        .why-no-trade {
          background: var(--bg-secondary, #1a1a2e);
          border-radius: 12px;
          padding: 20px;
          border: 1px solid var(--border-color, #333);
        }

        .why-no-trade--blocked {
          border-color: #ff4444;
        }

        .why-no-trade--approved {
          border-color: #44ff88;
        }

        .why-no-trade__header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 16px;
        }

        .why-no-trade__header h3 {
          margin: 0;
          font-size: 18px;
          color: var(--text-primary, #fff);
        }

        .why-no-trade__icon {
          font-size: 24px;
        }

        .why-no-trade__symbol {
          background: var(--bg-tertiary, #252542);
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
        }

        .why-no-trade__direction {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
        }

        .why-no-trade__direction--long {
          background: rgba(0, 255, 100, 0.2);
          color: #00ff64;
        }

        .why-no-trade__direction--short {
          background: rgba(255, 68, 68, 0.2);
          color: #ff4444;
        }

        .why-no-trade__timestamp {
          color: var(--text-secondary, #888);
          font-size: 12px;
          margin-bottom: 16px;
        }

        .why-no-trade__blocker {
          background: rgba(255, 68, 68, 0.1);
          border: 1px solid rgba(255, 68, 68, 0.3);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .why-no-trade__blocker-header {
          display: flex;
          gap: 12px;
          margin-bottom: 12px;
        }

        .why-no-trade__blocker-icon {
          font-size: 32px;
        }

        .why-no-trade__blocker-header h4 {
          margin: 0;
          color: #ff4444;
        }

        .why-no-trade__blocker-desc {
          margin: 4px 0 0;
          color: var(--text-secondary, #888);
          font-size: 12px;
        }

        .why-no-trade__blocker-reason {
          color: var(--text-primary, #fff);
          font-size: 14px;
        }

        .why-no-trade__blocker-details {
          margin-top: 12px;
        }

        .why-no-trade__blocker-details pre {
          background: var(--bg-tertiary, #252542);
          padding: 12px;
          border-radius: 4px;
          font-size: 11px;
          overflow-x: auto;
          color: var(--text-secondary, #888);
        }

        .why-no-trade__pipeline {
          margin-bottom: 20px;
        }

        .why-no-trade__pipeline h4 {
          margin: 0 0 12px;
          font-size: 14px;
          color: var(--text-secondary, #888);
        }

        .why-no-trade__gates {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .why-no-trade__gate {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          border-radius: 6px;
          font-size: 12px;
        }

        .why-no-trade__gate--passed {
          background: rgba(0, 255, 100, 0.1);
          border: 1px solid rgba(0, 255, 100, 0.3);
          color: #00ff64;
        }

        .why-no-trade__gate--blocked {
          background: rgba(255, 68, 68, 0.1);
          border: 1px solid rgba(255, 68, 68, 0.3);
          color: #ff4444;
        }

        .why-no-trade__gate-number {
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-tertiary, #252542);
          border-radius: 50%;
          font-size: 10px;
        }

        .why-no-trade__passed h4 {
          margin: 0 0 12px;
          font-size: 14px;
          color: var(--text-secondary, #888);
        }

        .why-no-trade__passed-list {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .why-no-trade__passed-badge {
          background: var(--bg-tertiary, #252542);
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
          color: var(--text-secondary, #888);
        }
      `}</style>
        </div>
    );
};

export default WhyNoTrade;
