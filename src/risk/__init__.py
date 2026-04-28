"""
Risk module - position sizing, limits, kill-switch
"""
from .advanced_sizing import VolatilityAdjustedSizing, KillSwitch
from .scalper_risk_engine import ScalperRiskConfig, ScalperRiskEngine
from .position_sizing import kelly_position_size, FractionalKellySizer
from .position_sizing import fixed_fractional_position_size
from .correlation import correlation_guard, correlation_matrix
from .drawdown_control import dynamic_risk_multiplier
from .var_calculator import value_at_risk, expected_shortfall
from .controls import RiskConfig, RiskCheckResult, RiskController

__all__ = [
	"VolatilityAdjustedSizing",
	"KillSwitch",
	"ScalperRiskConfig",
	"ScalperRiskEngine",
	"kelly_position_size",
	"FractionalKellySizer",
	"fixed_fractional_position_size",
	"correlation_guard",
	"correlation_matrix",
	"dynamic_risk_multiplier",
	"value_at_risk",
	"expected_shortfall",
	"RiskConfig",
	"RiskCheckResult",
	"RiskController",
]
