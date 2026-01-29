'use client';

import React from 'react';

/**
 * StartupProgress - v10.4 TRUST-GRADE
 * 
 * Visualizes system startup sequence.
 * Blocks operator view/interaction until system is READY.
 */

interface StartupStep {
    state: string;
    progress: number;
    details: string;
    error?: string;
}

interface StartupProgressProps {
    isVisible: boolean;
    currentState: string;
    totalProgress: number; // 0.0 to 1.0
    steps: Record<string, StartupStep>;
}

const StartupProgress: React.FC<StartupProgressProps> = ({
    isVisible,
    currentState,
    totalProgress,
    steps,
}) => {
    if (!isVisible) return null;

    return (
        <div className="startup-overlay">
            <div className="startup-card">
                <div className="startup-header">
                    <h2>System Initialization</h2>
                    <span className="startup-status-badge">
                        {currentState.toUpperCase().replace('_', ' ')}
                    </span>
                </div>

                {/* Main Progress Bar */}
                <div className="startup-progress-container">
                    <div
                        className="startup-progress-bar"
                        style={{ width: `${Math.min(100, Math.max(0, totalProgress * 100))}%` }}
                    />
                </div>
                <p className="startup-percentage">
                    {Math.round(totalProgress * 100)}% Complete
                </p>

                {/* Steps List */}
                <div className="startup-steps">
                    {Object.entries(steps).map(([name, step]) => (
                        <div key={name} className="startup-step">
                            <div className="startup-step-header">
                                <span className={`startup-step-icon ${getIconClass(step.state)}`}>
                                    {getIcon(step.state)}
                                </span>
                                <span className="startup-step-name">{name}</span>
                                {step.error && <span className="startup-step-error-badge">FAILED</span>}
                            </div>

                            <div className="startup-step-details">
                                {step.error ? (
                                    <span className="error-text">{step.error}</span>
                                ) : (
                                    <span>{step.details}</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                {currentState === 'failed' && (
                    <div className="startup-failure">
                        <h3>⚠️ Startup Failed</h3>
                        <p>Please check system logs. Manual restart required.</p>
                    </div>
                )}
            </div>

            <style jsx>{`
        .startup-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(10, 10, 20, 0.95);
          backdrop-filter: blur(10px);
          z-index: 9999;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .startup-card {
          width: 500px;
          background: #1a1a2e;
          border: 1px solid #333;
          border-radius: 12px;
          padding: 32px;
          box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }

        .startup-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }

        .startup-header h2 {
          margin: 0;
          font-size: 24px;
          color: #fff;
        }

        .startup-status-badge {
          background: #333;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: bold;
          color: #aaa;
        }

        .startup-progress-container {
          height: 8px;
          background: #222;
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 8px;
        }

        .startup-progress-bar {
          height: 100%;
          background: linear-gradient(90deg, #00aaff, #00ff88);
          transition: width 0.3s ease;
        }

        .startup-percentage {
          text-align: right;
          color: #888;
          font-size: 12px;
          margin-bottom: 24px;
        }

        .startup-steps {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .startup-step {
          background: #252542;
          padding: 12px;
          border-radius: 8px;
        }

        .startup-step-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 4px;
        }

        .startup-step-name {
          font-weight: 500;
          color: #ddd;
        }

        .startup-step-details {
          padding-left: 36px;
          font-size: 12px;
          color: #888;
        }

        .startup-step-icon {
          width: 24px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          background: #333;
          font-size: 14px;
        }

        .icon-success { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .icon-info { background: rgba(0, 170, 255, 0.2); color: #00aaff; }
        .icon-error { background: rgba(255, 68, 68, 0.2); color: #ff4444; }

        .error-text {
          color: #ff4444;
        }
        
        .startup-failure {
          margin-top: 24px;
          padding: 16px;
          background: rgba(255, 0, 0, 0.1);
          border: 1px solid #ff0000;
          border-radius: 8px;
          text-align: center;
        }
        
        .startup-failure h3 {
          color: #ff4444;
          margin: 0 0 8px 0;
        }
      `}</style>
        </div>
    );
};

function getIcon(state: string): string {
    if (state === 'ready_to_trade' || state === 'complete') return '✓';
    if (state === 'failed') return '✗';
    return '⋯';
}

function getIconClass(state: string): string {
    if (state === 'ready_to_trade' || state === 'complete') return 'icon-success';
    if (state === 'failed') return 'icon-error';
    return 'icon-info';
}

export default StartupProgress;
