'use client';

import React, { useState } from 'react';

/**
 * OperatorActions - v11.1 FINAL-MAP
 * 
 * Operator control panel with mandatory reason input.
 * All actions are logged and audited.
 */

interface OperatorActionsProps {
    tradingEnabled: boolean;
    currentMode: string;
    incidentState: string;
    onPauseTrading: (reason: string) => void;
    onResumeTrading: (reason: string) => void;
    onAcknowledgeIncident: (reason: string) => void;
}

const OperatorActions: React.FC<OperatorActionsProps> = ({
    tradingEnabled,
    currentMode,
    incidentState,
    onPauseTrading,
    onResumeTrading,
    onAcknowledgeIncident,
}) => {
    const [activeAction, setActiveAction] = useState<string | null>(null);
    const [reason, setReason] = useState('');
    const [confirmStep, setConfirmStep] = useState(false);

    const handleAction = () => {
        if (reason.length < 10) return;

        switch (activeAction) {
            case 'pause':
                onPauseTrading(reason);
                break;
            case 'resume':
                onResumeTrading(reason);
                break;
            case 'acknowledge':
                onAcknowledgeIncident(reason);
                break;
        }

        setActiveAction(null);
        setReason('');
        setConfirmStep(false);
    };

    const actions = [
        {
            id: 'pause',
            label: 'Pause Trading',
            icon: '⏸️',
            color: '#ffaa00',
            disabled: !tradingEnabled,
            show: true,
        },
        {
            id: 'resume',
            label: 'Resume Trading',
            icon: '▶️',
            color: '#00ff64',
            disabled: tradingEnabled || incidentState === 'HALTED',
            show: !tradingEnabled,
        },
        {
            id: 'acknowledge',
            label: 'Acknowledge Incident',
            icon: '✓',
            color: '#00aaff',
            disabled: incidentState === 'NORMAL',
            show: incidentState !== 'NORMAL',
        },
    ];

    return (
        <div className="operator-actions">
            <div className="operator-actions__header">
                <h3>Operator Controls</h3>
                <span className="operator-actions__mode" data-mode={currentMode}>
                    {currentMode.toUpperCase()}
                </span>
            </div>

            {!activeAction ? (
                <div className="operator-actions__list">
                    {actions
                        .filter((a) => a.show)
                        .map((action) => (
                            <button
                                key={action.id}
                                className="operator-actions__btn"
                                style={{ borderColor: action.color }}
                                disabled={action.disabled}
                                onClick={() => setActiveAction(action.id)}
                            >
                                <span>{action.icon}</span>
                                <span>{action.label}</span>
                            </button>
                        ))}
                </div>
            ) : (
                <div className="operator-actions__form">
                    <div className="operator-actions__form-header">
                        <span>{actions.find((a) => a.id === activeAction)?.icon}</span>
                        <span>{actions.find((a) => a.id === activeAction)?.label}</span>
                    </div>

                    {!confirmStep ? (
                        <>
                            <label>Reason (required, min 10 characters):</label>
                            <textarea
                                value={reason}
                                onChange={(e) => setReason(e.target.value)}
                                placeholder="Enter detailed reason for this action..."
                                rows={3}
                            />
                            <div className="operator-actions__form-buttons">
                                <button
                                    className="operator-actions__next"
                                    disabled={reason.length < 10}
                                    onClick={() => setConfirmStep(true)}
                                >
                                    Next →
                                </button>
                                <button
                                    className="operator-actions__cancel"
                                    onClick={() => {
                                        setActiveAction(null);
                                        setReason('');
                                    }}
                                >
                                    Cancel
                                </button>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="operator-actions__confirm">
                                <p>⚠️ Confirm this action:</p>
                                <p className="operator-actions__confirm-reason">"{reason}"</p>
                            </div>
                            <div className="operator-actions__form-buttons">
                                <button className="operator-actions__execute" onClick={handleAction}>
                                    Confirm & Execute
                                </button>
                                <button
                                    className="operator-actions__cancel"
                                    onClick={() => setConfirmStep(false)}
                                >
                                    Back
                                </button>
                            </div>
                        </>
                    )}
                </div>
            )}

            <div className="operator-actions__warning">
                <p>⚠️ All actions are logged and audited</p>
                <p>⚠️ No manual trade buttons available</p>
            </div>

            <style jsx>{`
        .operator-actions {
          background: var(--bg-secondary, #1a1a2e);
          border-radius: 12px;
          padding: 20px;
          border: 1px solid var(--border-color, #333);
        }

        .operator-actions__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .operator-actions__header h3 {
          margin: 0;
          font-size: 16px;
          color: var(--text-primary, #fff);
        }

        .operator-actions__mode {
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 700;
        }

        .operator-actions__mode[data-mode='live'] {
          background: rgba(255, 0, 0, 0.2);
          color: #ff4444;
        }

        .operator-actions__mode[data-mode='testnet'],
        .operator-actions__mode[data-mode='paper'] {
          background: rgba(100, 100, 100, 0.2);
          color: #888;
        }

        .operator-actions__list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .operator-actions__btn {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 12px 16px;
          background: transparent;
          border: 1px solid;
          border-radius: 8px;
          color: var(--text-primary, #fff);
          cursor: pointer;
          transition: all 0.2s;
        }

        .operator-actions__btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .operator-actions__btn:not(:disabled):hover {
          transform: translateX(4px);
        }

        .operator-actions__form {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .operator-actions__form-header {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 15px;
          font-weight: 600;
        }

        .operator-actions__form label {
          font-size: 13px;
          color: var(--text-secondary, #888);
        }

        .operator-actions__form textarea {
          width: 100%;
          padding: 10px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-tertiary, #0a0a0f);
          color: var(--text-primary, #fff);
          resize: none;
        }

        .operator-actions__form-buttons {
          display: flex;
          gap: 8px;
        }

        .operator-actions__next,
        .operator-actions__execute {
          flex: 1;
          padding: 10px;
          border: none;
          border-radius: 6px;
          background: #00aaff;
          color: #000;
          font-weight: 600;
          cursor: pointer;
        }

        .operator-actions__next:disabled {
          background: #555;
          cursor: not-allowed;
        }

        .operator-actions__execute {
          background: #00ff64;
        }

        .operator-actions__cancel {
          padding: 10px 16px;
          border: 1px solid #555;
          border-radius: 6px;
          background: transparent;
          color: #888;
          cursor: pointer;
        }

        .operator-actions__confirm {
          padding: 12px;
          background: rgba(255, 170, 0, 0.1);
          border-radius: 8px;
        }

        .operator-actions__confirm p {
          margin: 0;
        }

        .operator-actions__confirm-reason {
          font-style: italic;
          color: var(--text-secondary, #aaa);
          margin-top: 8px !important;
        }

        .operator-actions__warning {
          margin-top: 16px;
          padding-top: 12px;
          border-top: 1px solid var(--border-color, #333);
        }

        .operator-actions__warning p {
          margin: 4px 0;
          font-size: 11px;
          color: #888;
        }
      `}</style>
        </div>
    );
};

export default OperatorActions;
