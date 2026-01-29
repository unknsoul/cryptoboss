'use client';

import React from 'react';

/**
 * EnvironmentBanner - v10.4 TRUST-GRADE
 * 
 * Sticky banner showing current environment (LIVE/TESTNET/PAPER).
 * Ensures operator never forgets which mode they are in.
 */

type EnvMode = 'LIVE' | 'TESTNET' | 'PAPER';

interface EnvironmentBannerProps {
    mode: string; // "live", "testnet", "paper"
    exchangeName?: string;
    isSimulated?: boolean;
}

const EnvironmentBanner: React.FC<EnvironmentBannerProps> = ({
    mode,
    exchangeName = 'Unknown',
    isSimulated = false,
}) => {
    const normalizedMode = mode.toUpperCase() as EnvMode;

    const config = {
        LIVE: {
            color: '#ff0000',
            bg: '#330000',
            border: '#ff0000',
            label: '🔴 LIVE TRADING - REAL CAPITAL',
            warning: 'Actions are final and execute on real exchange.',
        },
        TESTNET: {
            color: '#ffaa00',
            bg: '#332200',
            border: '#ffaa00',
            label: '🟡 TESTNET ENVIRONMENT',
            warning: 'Connected to exchange testnet. Virtual funds.',
        },
        PAPER: {
            color: '#00aaff',
            bg: '#001133',
            border: '#00aaff',
            label: '🔵 PAPER TRADING (SIMULATED)',
            warning: 'Internal simulation. No exchange connection.',
        },
    }[normalizedMode] || {
        color: '#888888',
        bg: '#222222',
        border: '#444444',
        label: '⚪ UNKNOWN MODE',
        warning: 'System environment unverified.',
    };

    return (
        <div className="env-banner">
            <div className="env-banner__content">
                <div className="env-banner__left">
                    <span className="env-banner__badge" style={{ borderColor: config.border, color: config.color, background: `${config.color}22` }}>
                        {normalizedMode}
                    </span>
                    <span className="env-banner__exchange">
                        {exchangeName.toUpperCase()} {isSimulated ? '(SIMULATED)' : ''}
                    </span>
                </div>

                <div className="env-banner__center">
                    <span className="env-banner__label" style={{ color: config.color }}>
                        {config.label}
                    </span>
                </div>

                <div className="env-banner__right">
                    <span className="env-banner__warning">
                        {config.warning}
                    </span>
                </div>
            </div>

            <style jsx>{`
        .env-banner {
          position: sticky;
          top: 0;
          z-index: 1000;
          background: ${config.bg};
          border-bottom: 2px solid ${config.border};
          padding: 8px 16px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }

        .env-banner__content {
          display: flex;
          justify-content: space-between;
          align-items: center;
          max-width: 1400px;
          margin: 0 auto;
        }

        .env-banner__left, .env-banner__right {
          flex: 1;
          display: flex;
          align-items: center;
        }

        .env-banner__right {
          justify-content: flex-end;
          text-align: right;
        }

        .env-banner__center {
           flex: 2;
           text-align: center;
        }

        .env-banner__badge {
          font-family: monospace;
          font-weight: bold;
          font-size: 12px;
          border: 1px solid;
          padding: 2px 6px;
          border-radius: 4px;
          margin-right: 12px;
        }

        .env-banner__exchange {
          color: #aaa;
          font-size: 12px;
          font-weight: 500;
        }

        .env-banner__label {
          font-weight: 800;
          letter-spacing: 1px;
          font-size: 14px;
          text-transform: uppercase;
          animation: ${normalizedMode === 'LIVE' ? 'pulse 2s infinite' : 'none'};
        }

        .env-banner__warning {
          color: #bbb;
          font-size: 11px;
          font-style: italic;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; text-shadow: 0 0 10px ${config.color}; }
          50% { opacity: 0.8; text-shadow: 0 0 2px ${config.color}; }
        }
      `}</style>
        </div>
    );
};

export default EnvironmentBanner;
