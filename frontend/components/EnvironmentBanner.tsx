'use client';

import React from 'react';

/**
 * EnvironmentBanner - CRYPTOBOSS 2.0
 * 
 * PAPER TRADING REMOVED.
 * Only TESTNET and LIVE environments are supported.
 * 
 * Sticky banner showing current environment.
 * Ensures operator never forgets which mode they are in.
 */

type EnvMode = 'LIVE' | 'TESTNET';

interface EnvironmentBannerProps {
  mode: string; // "live" or "testnet"
  exchangeName?: string;
}

const EnvironmentBanner: React.FC<EnvironmentBannerProps> = ({
  mode,
  exchangeName = 'Binance',
}) => {
  // Normalize mode - paper defaults to testnet
  let normalizedMode: EnvMode = mode.toUpperCase() as EnvMode;
  if (normalizedMode !== 'LIVE' && normalizedMode !== 'TESTNET') {
    normalizedMode = 'TESTNET'; // Default to testnet, never paper
  }

  const config = {
    LIVE: {
      color: '#ff0000',
      bg: '#330000',
      border: '#ff0000',
      label: '🔴 LIVE TRADING - REAL CAPITAL AT RISK',
      warning: 'Actions are final and execute on REAL exchange.',
      pulse: true,
    },
    TESTNET: {
      color: '#ffaa00',
      bg: '#332200',
      border: '#ffaa00',
      label: '🟡 TESTNET ENVIRONMENT',
      warning: 'Connected to Binance Testnet. Virtual funds only.',
      pulse: false,
    },
  }[normalizedMode];

  return (
    <div className="env-banner">
      <div className="env-banner__content">
        <div className="env-banner__left">
          <span
            className="env-banner__badge"
            style={{
              borderColor: config.border,
              color: config.color,
              background: `${config.color}22`
            }}
          >
            {normalizedMode}
          </span>
          <span className="env-banner__exchange">
            {exchangeName.toUpperCase()}
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
                    animation: ${config.pulse ? 'pulse 2s infinite' : 'none'};
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
