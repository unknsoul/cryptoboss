"""CryptoBoss v4 API route blueprint.

This module contains route snippets intended to be merged into dashboard/api.py.
"""

V4_API_ROUTE_SNIPPET = r'''
from src.v3.orchestrator_v4 import OrchestratorV4
from src.v3.config_v4 import V4SystemConfig
from src.strategies.pro_strategy_builder import ProStrategyBuilder, INDICATOR_LIBRARY

_v4 = None

def get_v4():
    global _v4
    if _v4 is None:
        _v4 = OrchestratorV4()
    return _v4

@app.get("/api/v4/status")
async def v4_status():
    return get_v4().status()

@app.get("/api/v4/config")
async def v4_config():
    return get_v4().config.summary()
'''
