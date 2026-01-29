'use client';

import React, { useMemo } from 'react';

/**
 * IncidentTimeline - v10.3 Explainability
 * 
 * Visual timeline of system state changes and incidents.
 * Shows transitions between NORMAL, DEGRADED, FREEZE, and HALTED states.
 */

interface IncidentEvent {
    timestamp: string;
    from_state: string;
    to_state: string;
    reason: string;
    triggered_by: string;
    auto_recoverable: boolean;
    context?: Record<string, unknown>;
}

interface IncidentTimelineProps {
    events: IncidentEvent[];
    currentState: string;
    requiresAcknowledgment?: boolean;
}

const stateConfig: Record<string, { color: string; icon: string; label: string }> = {
    normal: { color: '#00ff64', icon: '✅', label: 'Normal' },
    degraded: { color: '#ffaa00', icon: '⚠️', label: 'Degraded' },
    incident_freeze: { color: '#ff4444', icon: '🧊', label: 'Frozen' },
    halted: { color: '#ff0000', icon: '🛑', label: 'Halted' },
};

const IncidentTimeline: React.FC<IncidentTimelineProps> = ({
    events,
    currentState,
    requiresAcknowledgment,
}) => {
    const sortedEvents = useMemo(() => {
        return [...events].sort(
            (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
    }, [events]);

    const currentStateConfig = stateConfig[currentState] || stateConfig.normal;

    return (
        <div className="incident-timeline">
            {/* Current State Banner */}
            <div
                className="incident-timeline__current"
                style={{ borderColor: currentStateConfig.color }}
            >
                <div className="incident-timeline__current-header">
                    <span className="incident-timeline__current-icon">
                        {currentStateConfig.icon}
                    </span>
                    <div>
                        <h3>Current State: {currentStateConfig.label}</h3>
                        {requiresAcknowledgment && (
                            <span className="incident-timeline__ack-badge">
                                ⚠️ Acknowledgment Required
                            </span>
                        )}
                    </div>
                </div>
                {currentState !== 'normal' && (
                    <p className="incident-timeline__current-warning">
                        {currentState === 'degraded'
                            ? 'Trading at reduced capacity. Some strategies may be limited.'
                            : currentState === 'incident_freeze'
                                ? 'Trading frozen. Reduce-only mode active. Operator intervention required.'
                                : 'System halted. No trading activity permitted.'}
                    </p>
                )}
            </div>

            {/* Timeline */}
            <div className="incident-timeline__list">
                <h4>Incident History</h4>
                {sortedEvents.length === 0 ? (
                    <p className="incident-timeline__empty">No incidents recorded</p>
                ) : (
                    <div className="incident-timeline__events">
                        {sortedEvents.map((event, idx) => (
                            <div key={idx} className="incident-timeline__event">
                                <div className="incident-timeline__event-connector" />
                                <div
                                    className="incident-timeline__event-dot"
                                    style={{
                                        backgroundColor:
                                            stateConfig[event.to_state]?.color || '#888',
                                    }}
                                />
                                <div className="incident-timeline__event-content">
                                    <div className="incident-timeline__event-header">
                                        <span className="incident-timeline__event-time">
                                            {new Date(event.timestamp).toLocaleString()}
                                        </span>
                                        <div className="incident-timeline__event-states">
                                            <span
                                                className="incident-timeline__state-badge"
                                                style={{
                                                    backgroundColor: `${stateConfig[event.from_state]?.color || '#888'}22`,
                                                    color: stateConfig[event.from_state]?.color || '#888',
                                                }}
                                            >
                                                {stateConfig[event.from_state]?.label || event.from_state}
                                            </span>
                                            <span className="incident-timeline__arrow">→</span>
                                            <span
                                                className="incident-timeline__state-badge"
                                                style={{
                                                    backgroundColor: `${stateConfig[event.to_state]?.color || '#888'}22`,
                                                    color: stateConfig[event.to_state]?.color || '#888',
                                                }}
                                            >
                                                {stateConfig[event.to_state]?.label || event.to_state}
                                            </span>
                                        </div>
                                    </div>
                                    <p className="incident-timeline__event-reason">
                                        {event.reason}
                                    </p>
                                    <div className="incident-timeline__event-meta">
                                        <span>By: {event.triggered_by}</span>
                                        {event.auto_recoverable && (
                                            <span className="incident-timeline__auto-badge">
                                                Auto-recoverable
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <style jsx>{`
        .incident-timeline {
          background: var(--bg-secondary, #1a1a2e);
          border-radius: 12px;
          padding: 20px;
          border: 1px solid var(--border-color, #333);
        }

        .incident-timeline__current {
          background: var(--bg-tertiary, #252542);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
          border-left: 4px solid;
        }

        .incident-timeline__current-header {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .incident-timeline__current-icon {
          font-size: 32px;
        }

        .incident-timeline__current-header h3 {
          margin: 0;
          font-size: 18px;
          color: var(--text-primary, #fff);
        }

        .incident-timeline__ack-badge {
          display: inline-block;
          background: rgba(255, 170, 0, 0.2);
          color: #ffaa00;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          margin-top: 4px;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }

        .incident-timeline__current-warning {
          margin: 12px 0 0;
          color: var(--text-secondary, #888);
          font-size: 14px;
        }

        .incident-timeline__list h4 {
          margin: 0 0 16px;
          color: var(--text-secondary, #888);
          font-size: 14px;
        }

        .incident-timeline__empty {
          color: var(--text-secondary, #888);
          font-size: 14px;
          text-align: center;
          padding: 20px;
        }

        .incident-timeline__events {
          position: relative;
        }

        .incident-timeline__event {
          display: flex;
          position: relative;
          padding-left: 24px;
          padding-bottom: 20px;
        }

        .incident-timeline__event:last-child {
          padding-bottom: 0;
        }

        .incident-timeline__event-connector {
          position: absolute;
          left: 7px;
          top: 16px;
          bottom: 0;
          width: 2px;
          background: var(--border-color, #333);
        }

        .incident-timeline__event:last-child .incident-timeline__event-connector {
          display: none;
        }

        .incident-timeline__event-dot {
          position: absolute;
          left: 0;
          top: 4px;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          border: 2px solid var(--bg-secondary, #1a1a2e);
        }

        .incident-timeline__event-content {
          flex: 1;
          background: var(--bg-tertiary, #252542);
          border-radius: 8px;
          padding: 12px;
        }

        .incident-timeline__event-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 8px;
        }

        .incident-timeline__event-time {
          color: var(--text-secondary, #888);
          font-size: 12px;
        }

        .incident-timeline__event-states {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .incident-timeline__state-badge {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 600;
        }

        .incident-timeline__arrow {
          color: var(--text-secondary, #888);
        }

        .incident-timeline__event-reason {
          margin: 0 0 8px;
          color: var(--text-primary, #fff);
          font-size: 14px;
        }

        .incident-timeline__event-meta {
          display: flex;
          gap: 12px;
          font-size: 12px;
          color: var(--text-secondary, #888);
        }

        .incident-timeline__auto-badge {
          background: rgba(0, 255, 100, 0.1);
          color: #00ff64;
          padding: 2px 6px;
          border-radius: 4px;
        }
      `}</style>
        </div>
    );
};

export default IncidentTimeline;
