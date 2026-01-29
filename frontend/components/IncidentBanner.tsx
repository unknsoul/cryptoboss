'use client';

import React, { useState } from 'react';

/**
 * IncidentBanner - v11.1 FINAL-MAP
 * 
 * Blocks trading UI during incident states.
 * Requires operator acknowledgment to proceed.
 */

interface IncidentBannerProps {
    incidentState: 'NORMAL' | 'DEGRADED' | 'INCIDENT_FREEZE' | 'HALTED';
    reason: string;
    since: string;
    requiresAcknowledgment: boolean;
    onAcknowledge?: (reason: string) => void;
}

const IncidentBanner: React.FC<IncidentBannerProps> = ({
    incidentState,
    reason,
    since,
    requiresAcknowledgment,
    onAcknowledge,
}) => {
    const [ackReason, setAckReason] = useState('');
    const [showAckForm, setShowAckForm] = useState(false);

    if (incidentState === 'NORMAL') return null;

    const config = {
        DEGRADED: {
            icon: '⚠️',
            color: '#ffaa00',
            bg: '#332200',
            label: 'SYSTEM DEGRADED',
            message: 'Trading at reduced capacity. Some features may be limited.',
            blocking: false,
        },
        INCIDENT_FREEZE: {
            icon: '🧊',
            color: '#00aaff',
            bg: '#001133',
            label: 'TRADING FROZEN',
            message: 'New trades blocked. Position reduction only. Operator acknowledgment required.',
            blocking: true,
        },
        HALTED: {
            icon: '🛑',
            color: '#ff0000',
            bg: '#330000',
            label: 'SYSTEM HALTED',
            message: 'All trading suspended. Manual intervention required.',
            blocking: true,
        },
    }[incidentState];

    const handleAcknowledge = () => {
        if (ackReason.length >= 10 && onAcknowledge) {
            onAcknowledge(ackReason);
            setAckReason('');
            setShowAckForm(false);
        }
    };

    return (
        <div className="incident-banner" style={{ background: config.bg, borderColor: config.color }}>
            <div className="incident-banner__content">
                <div className="incident-banner__header">
                    <span className="incident-banner__icon">{config.icon}</span>
                    <h2 style={{ color: config.color }}>{config.label}</h2>
                </div>

                <p className="incident-banner__message">{config.message}</p>

                <div className="incident-banner__details">
                    <div className="incident-banner__detail">
                        <span>Reason:</span>
                        <span>{reason}</span>
                    </div>
                    <div className="incident-banner__detail">
                        <span>Since:</span>
                        <span>{new Date(since).toLocaleString()}</span>
                    </div>
                </div>

                {requiresAcknowledgment && (
                    <div className="incident-banner__ack">
                        {!showAckForm ? (
                            <button
                                className="incident-banner__ack-btn"
                                onClick={() => setShowAckForm(true)}
                                style={{ borderColor: config.color, color: config.color }}
                            >
                                Acknowledge Incident
                            </button>
                        ) : (
                            <div className="incident-banner__ack-form">
                                <input
                                    type="text"
                                    placeholder="Enter reason for acknowledgment (min 10 chars)..."
                                    value={ackReason}
                                    onChange={(e) => setAckReason(e.target.value)}
                                    className="incident-banner__ack-input"
                                />
                                <button
                                    onClick={handleAcknowledge}
                                    disabled={ackReason.length < 10}
                                    className="incident-banner__ack-submit"
                                    style={{ background: ackReason.length >= 10 ? config.color : '#555' }}
                                >
                                    Confirm
                                </button>
                                <button
                                    onClick={() => setShowAckForm(false)}
                                    className="incident-banner__ack-cancel"
                                >
                                    Cancel
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {config.blocking && <div className="incident-banner__overlay" />}

            <style jsx>{`
        .incident-banner {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 9998;
          border-bottom: 2px solid;
          padding: 20px;
        }

        .incident-banner__overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.7);
          z-index: -1;
          pointer-events: all;
        }

        .incident-banner__content {
          max-width: 800px;
          margin: 0 auto;
          text-align: center;
        }

        .incident-banner__header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          margin-bottom: 12px;
        }

        .incident-banner__icon {
          font-size: 36px;
        }

        .incident-banner__header h2 {
          margin: 0;
          font-size: 24px;
          font-weight: 800;
          letter-spacing: 2px;
        }

        .incident-banner__message {
          color: #ddd;
          font-size: 16px;
          margin-bottom: 16px;
        }

        .incident-banner__details {
          display: flex;
          justify-content: center;
          gap: 32px;
          margin-bottom: 20px;
        }

        .incident-banner__detail {
          display: flex;
          gap: 8px;
          font-size: 14px;
        }

        .incident-banner__detail span:first-child {
          color: #888;
        }

        .incident-banner__detail span:last-child {
          color: #fff;
        }

        .incident-banner__ack {
          margin-top: 16px;
        }

        .incident-banner__ack-btn {
          background: transparent;
          border: 2px solid;
          padding: 12px 24px;
          border-radius: 8px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .incident-banner__ack-btn:hover {
          transform: scale(1.02);
        }

        .incident-banner__ack-form {
          display: flex;
          gap: 8px;
          justify-content: center;
          flex-wrap: wrap;
        }

        .incident-banner__ack-input {
          width: 300px;
          padding: 10px 14px;
          border-radius: 6px;
          border: 1px solid #444;
          background: #1a1a2e;
          color: #fff;
          font-size: 14px;
        }

        .incident-banner__ack-submit {
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          color: #000;
          font-weight: 600;
          cursor: pointer;
        }

        .incident-banner__ack-cancel {
          padding: 10px 20px;
          border: 1px solid #555;
          border-radius: 6px;
          background: transparent;
          color: #888;
          cursor: pointer;
        }
      `}</style>
        </div>
    );
};

export default IncidentBanner;
