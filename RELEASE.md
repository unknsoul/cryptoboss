# CryptoBoss v1.0.0 - Final Release Specification

> **Release Status**: FINAL_PERSONAL_LIVE  
> **Intended Use**: Personal trading system with future startup potential  
> **Release Policy**: No new features after this release

---

## Core Philosophy

### Principles
1. **Survival over profit** — Capital preservation is the priority
2. **No forced trades** — System waits for valid conditions
3. **Explainability over prediction** — Every decision has a reason
4. **Risk veto overrides everything** — No bypass possible
5. **Human is responsible, system is strict** — Operator accountable

### Anti-Goals
- ❌ No get-rich-quick behavior
- ❌ No prediction-driven trading
- ❌ No emotional overrides
- ❌ No opaque automation

---

## Environment & Safety

### Modes
| Mode | Exchange URL | Visual Theme |
|------|--------------|--------------|
| TESTNET | testnet.binance.vision | Neutral/Gray |
| LIVE | api.binance.com | Red Warning |

### Environment Rules
- Environment selected **only at startup**
- Environment is **immutable during runtime**
- Each API response contains `environment_signature`
- Frontend blocks rendering on signature mismatch
- LIVE mode uses permanent visual warning theme

### Exchange Safety
- Testnet and live endpoints fully isolated
- Timestamp drift detection mandatory
- Partial API responses treated as degraded
- API key revocation triggers incident freeze

---

## Decision Architecture

```
MarketContextEngine → ContextStateMachine → BiasEngine
          ↓
    StrategyProposalLayer → ScoringContract
          ↓
    TradePermissionFilter → RiskGuardian → CapitalGovernor
          ↓
      ExecutionRouter
```

### Hard Rules
- **No step can be skipped**
- **Any veto halts the pipeline**
- **Strategies never execute trades**
- **All decisions must be explainable**

---

## Market Context & Bias

### Market Contexts
| Context | Description | Capital |
|---------|-------------|---------|
| TRENDING_UP | Strong uptrend | 100% |
| TRENDING_DOWN | Strong downtrend | 100% |
| RANGING | Sideways market | 75% |
| HIGH_VOLATILITY | Unstable conditions | 30% |
| NO_TRADE | Unclear/dangerous | 0% |

### Bias Rules
- Bias flip cooldown enforced
- Trades against bias forbidden
- Neutral bias reduces capital allocation

---

## Strategy & Scoring

### Strategy Constraints
- Strategies may only return `EntryProposal`
- Strategies cannot access balances or execution
- Each proposal must include reasoning

### Scoring Contract
| Component | Weight |
|-----------|--------|
| `context_fit` | 30% |
| `historical_expectancy` | 25% |
| `recent_performance_decay` | 25% |
| `risk_alignment` | 20% |

**Rules**: All components required. Invalid proposals rejected silently.

---

## Risk & Capital Management

### Risk Guardian Controls
| Control | Description |
|---------|-------------|
| `max_daily_drawdown` | Daily loss limit |
| `max_exposure` | Maximum position value |
| `max_consecutive_losses` | Circuit breaker |
| `kill_switch` | Emergency halt (persists across restarts) |

### Capital Governor
- Automatic allocation based on context
- Modifiers: exchange health, drawdown, bias strength
- NO_TRADE = 0% allocation

---

## Incident Handling

| State | New Trades | Reduce Positions | Exit Required |
|-------|------------|------------------|---------------|
| NORMAL | ✅ | ✅ | — |
| DEGRADED | ❌ | ✅ | — |
| INCIDENT_FREEZE | ❌ | ✅ | Operator ACK |
| HALTED | ❌ | ❌ | Manual intervention |

---

## Data Authenticity

| Tag | Meaning | UI Treatment |
|-----|---------|--------------|
| LIVE_EXCHANGE | Real-time exchange data | Normal |
| TESTNET_EXCHANGE | Testnet data | Normal (gray) |
| DERIVED | Calculated values | Gray label |
| SIMULATED | Paper trading | Dashed border |
| STALE | Old data | Yellow warning |

---

## Decision Explainability

### Mandatory Outputs
- `why_trade_allowed` — Approval explanation
- `why_trade_blocked` — Veto explanation
- `why_no_trade` — Market condition explanation
- `which_gate_failed` — Specific gate reference
- `why_position_size_changed` — Sizing explanation

**Rule**: No silent vetoes. All narratives stored.

---

## Operator Control

### Allowed Actions
| Action | Requires Reason | Logged |
|--------|-----------------|--------|
| Pause Trading | ✅ | ✅ |
| Resume Trading | ✅ | ✅ |
| Acknowledge Incident | ✅ | ✅ |

**Constraint**: Operator cannot bypass risk or capital veto.

---

## Frontend UI

### Global Elements
- Environment Badge (TESTNET / LIVE)
- Exchange Health Indicator
- Incident State Banner

### Core Screens
1. **Dashboard**: System status, Why No Trade, Gate breakdown
2. **Positions**: Entry reason, data tag, exit reason
3. **Incident Timeline**: All state changes
4. **Operator Controls**: Pause, resume, acknowledge

---

## Testing Requirements

**Priority**: Failure paths over success paths

### Mandatory Tests
- `timestamp_out_of_sync`
- `partial_balance_response`
- `api_key_revoked_mid_session`
- `restart_with_open_positions`
- `exchange_health_flapping`
- `config_checksum_mismatch`

---

## Known Limitations (Accepted)

| Limitation | Acceptance |
|------------|------------|
| No guaranteed profitability | Expected |
| Long no-trade periods | By design |
| Performance varies by regime | Understood |
| Not for high-frequency trading | Out of scope |

### Explicitly Deferred
- AI-driven trade execution
- Auto-optimization
- Multi-exchange routing
- Retail user onboarding

---

## Definition of Done

| Requirement | Status |
|-------------|--------|
| Environment is never ambiguous | ✅ |
| Displayed data is always trustworthy | ✅ |
| Every decision is explainable | ✅ |
| Incident handling is unavoidable | ✅ |
| System fails safe under stress | ✅ |

---

## Post-Release Rules

1. **No architecture refactors**
2. **No new strategies**
3. **No leverage increase**
4. **No emotional overrides**
5. **Observe for months before any expansion**

---

**Version**: 1.0.0  
**Release Date**: January 2026  
**Status**: FINAL_PERSONAL_LIVE
