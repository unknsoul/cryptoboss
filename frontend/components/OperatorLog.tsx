'use client';

import React, { useMemo, useState } from 'react';

/**
 * OperatorLog - v10.3 Explainability
 * 
 * Table view of all operator interventions and their reasons.
 * Shows audit trail with identity, action, reason, and timestamps.
 */

interface OperatorIntervention {
    record_id: string;
    timestamp: string;
    operator: {
        operator_id: string;
        name: string;
        role: string;
    };
    action_type: string;
    reason: {
        code: string;
        description: string;
    };
    success: boolean;
    error_message?: string;
}

interface OperatorLogProps {
    interventions: OperatorIntervention[];
    maxItems?: number;
}

const actionLabels: Record<string, { label: string; icon: string; color: string }> = {
    pause_trading: { label: 'Pause Trading', icon: '⏸️', color: '#ffaa00' },
    resume_trading: { label: 'Resume Trading', icon: '▶️', color: '#00ff64' },
    halt_system: { label: 'Halt System', icon: '🛑', color: '#ff0000' },
    acknowledge_incident: { label: 'Acknowledge', icon: '✅', color: '#00aaff' },
    modify_config: { label: 'Modify Config', icon: '⚙️', color: '#aa88ff' },
    force_reduce_position: { label: 'Force Reduce', icon: '📉', color: '#ff8844' },
    emergency_close_all: { label: 'Emergency Close', icon: '🚨', color: '#ff0000' },
    clear_freeze: { label: 'Clear Freeze', icon: '🔓', color: '#00ff64' },
};

const roleColors: Record<string, string> = {
    admin: '#ff88aa',
    trader: '#88aaff',
    risk_manager: '#ffaa88',
};

const OperatorLog: React.FC<OperatorLogProps> = ({
    interventions,
    maxItems = 50,
}) => {
    const [filter, setFilter] = useState<string>('all');

    const filteredInterventions = useMemo(() => {
        let result = [...interventions];

        if (filter !== 'all') {
            result = result.filter(i => i.action_type === filter);
        }

        return result
            .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
            .slice(0, maxItems);
    }, [interventions, filter, maxItems]);

    const uniqueActions = useMemo(() => {
        return Array.from(new Set(interventions.map(i => i.action_type)));
    }, [interventions]);

    return (
        <div className="operator-log">
            <div className="operator-log__header">
                <h3>📋 Operator Actions Log</h3>
                <div className="operator-log__filter">
                    <select
                        value={filter}
                        onChange={e => setFilter(e.target.value)}
                        className="operator-log__select"
                    >
                        <option value="all">All Actions</option>
                        {uniqueActions.map(action => (
                            <option key={action} value={action}>
                                {actionLabels[action]?.label || action}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {filteredInterventions.length === 0 ? (
                <div className="operator-log__empty">
                    <p>No operator interventions recorded</p>
                </div>
            ) : (
                <div className="operator-log__table-container">
                    <table className="operator-log__table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Operator</th>
                                <th>Action</th>
                                <th>Reason</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredInterventions.map(intervention => {
                                const actionConfig = actionLabels[intervention.action_type] || {
                                    label: intervention.action_type,
                                    icon: '📝',
                                    color: '#888',
                                };

                                return (
                                    <tr key={intervention.record_id}>
                                        <td className="operator-log__cell-time">
                                            {new Date(intervention.timestamp).toLocaleString()}
                                        </td>
                                        <td className="operator-log__cell-operator">
                                            <div className="operator-log__operator">
                                                <span className="operator-log__operator-name">
                                                    {intervention.operator.name}
                                                </span>
                                                <span
                                                    className="operator-log__operator-role"
                                                    style={{
                                                        color: roleColors[intervention.operator.role] || '#888',
                                                    }}
                                                >
                                                    {intervention.operator.role}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="operator-log__cell-action">
                                            <span
                                                className="operator-log__action-badge"
                                                style={{
                                                    backgroundColor: `${actionConfig.color}22`,
                                                    color: actionConfig.color,
                                                }}
                                            >
                                                {actionConfig.icon} {actionConfig.label}
                                            </span>
                                        </td>
                                        <td className="operator-log__cell-reason">
                                            <div className="operator-log__reason">
                                                <span className="operator-log__reason-code">
                                                    {intervention.reason.code.replace(/_/g, ' ')}
                                                </span>
                                                <span className="operator-log__reason-desc">
                                                    {intervention.reason.description}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="operator-log__cell-status">
                                            {intervention.success ? (
                                                <span className="operator-log__status operator-log__status--success">
                                                    ✓ Success
                                                </span>
                                            ) : (
                                                <span className="operator-log__status operator-log__status--failed">
                                                    ✗ Failed
                                                    {intervention.error_message && (
                                                        <span className="operator-log__error">
                                                            {intervention.error_message}
                                                        </span>
                                                    )}
                                                </span>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}

            <div className="operator-log__footer">
                <span>
                    Showing {filteredInterventions.length} of {interventions.length} interventions
                </span>
            </div>

            <style jsx>{`
        .operator-log {
          background: var(--bg-secondary, #1a1a2e);
          border-radius: 12px;
          padding: 20px;
          border: 1px solid var(--border-color, #333);
        }

        .operator-log__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .operator-log__header h3 {
          margin: 0;
          font-size: 18px;
          color: var(--text-primary, #fff);
        }

        .operator-log__select {
          background: var(--bg-tertiary, #252542);
          border: 1px solid var(--border-color, #333);
          border-radius: 6px;
          padding: 8px 12px;
          color: var(--text-primary, #fff);
          font-size: 14px;
        }

        .operator-log__empty {
          text-align: center;
          padding: 40px;
          color: var(--text-secondary, #888);
        }

        .operator-log__table-container {
          overflow-x: auto;
        }

        .operator-log__table {
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }

        .operator-log__table th {
          text-align: left;
          padding: 12px;
          background: var(--bg-tertiary, #252542);
          color: var(--text-secondary, #888);
          font-weight: 600;
          font-size: 12px;
          text-transform: uppercase;
        }

        .operator-log__table td {
          padding: 12px;
          border-bottom: 1px solid var(--border-color, #333);
          vertical-align: top;
        }

        .operator-log__cell-time {
          color: var(--text-secondary, #888);
          font-size: 12px;
          white-space: nowrap;
        }

        .operator-log__operator {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .operator-log__operator-name {
          color: var(--text-primary, #fff);
          font-weight: 500;
        }

        .operator-log__operator-role {
          font-size: 11px;
          text-transform: capitalize;
        }

        .operator-log__action-badge {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 6px;
          font-size: 12px;
          font-weight: 500;
          white-space: nowrap;
        }

        .operator-log__reason {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .operator-log__reason-code {
          color: var(--text-secondary, #888);
          font-size: 11px;
          text-transform: capitalize;
        }

        .operator-log__reason-desc {
          color: var(--text-primary, #fff);
          font-size: 13px;
        }

        .operator-log__status {
          font-size: 12px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .operator-log__status--success {
          color: #00ff64;
        }

        .operator-log__status--failed {
          color: #ff4444;
        }

        .operator-log__error {
          font-size: 11px;
          color: var(--text-secondary, #888);
        }

        .operator-log__footer {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid var(--border-color, #333);
          text-align: center;
          color: var(--text-secondary, #888);
          font-size: 12px;
        }
      `}</style>
        </div>
    );
};

export default OperatorLog;
