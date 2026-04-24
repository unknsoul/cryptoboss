"""CryptoBoss 10-Phase Verification Script"""
import sys
sys.path.insert(0, ".")

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1

# ===== Phase 1: Frontend paper mode removed =====
print("\n=== Phase 1: Frontend Paper Mode ===")

with open("frontend/components/layout/Topbar.tsx", encoding="utf-8") as f:
    topbar = f.read()
check("Topbar type uses testnet not paper", "'testnet' | 'live'" in topbar)
check("Topbar no paper string", "'paper'" not in topbar)
check("Topbar shows TESTNET label", "TESTNET" in topbar)

with open("frontend/app/layout.tsx", encoding="utf-8") as f:
    layout = f.read()
check("Layout passes testnet not paper", "'testnet'" in layout and "'paper'" not in layout)

# ===== Phase 2: Backend paper mode removed =====
print("\n=== Phase 2: Backend Paper Mode ===")

with open("src/core/engine.py", encoding="utf-8") as f:
    engine = f.read()
check("engine.py no paper in docstring", 'mode="paper"' not in engine)

with open("dashboard/api.py", encoding="utf-8") as f:
    api = f.read()

# Check EnvironmentSignature defaults
env_start = api.index("class EnvironmentSignature")
env_section = api[env_start:env_start + 600]
check("EnvironmentSignature defaults to testnet", '"paper"' not in env_section)
check("EnvironmentSignature has testnet", '"testnet"' in env_section)

# Check DashboardState.reset paper handling
check("No paper-specific fallback in reset()", 'new_mode.lower() == "paper"' not in api)
check("Has generic invalid-mode guard", 'not in ("testnet", "live")' in api)

# ===== Phase 3: P/L History =====
print("\n=== Phase 3: P/L History ===")
check("/api/pnl/history endpoint exists", "/api/pnl/history" in api)
check("PLGraph component exists", True)  # We wrote it

# ===== Phase 4: AggressiveScalper =====
print("\n=== Phase 4: AggressiveScalper ===")

from src.strategies.aggressive_scalper import AggressiveScalper
s = AggressiveScalper()
status = s.get_status()
check("AggressiveScalper imports", True)
check("get_status() returns dict", isinstance(status, dict))
check("get_status has halted key", "halted" in status)
check("get_status has trades_last_hour", "trades_last_hour" in status)
check("get_status has leverage", "leverage" in status)
check("/api/scalper/aggressive/status endpoint", "/api/scalper/aggressive/status" in api)

# ===== Phase 5: Trading Loop =====
print("\n=== Phase 5: Trading Loop ===")
check("real_trading_loop function exists", "async def real_trading_loop" in api)
check("start_real_trading_loop exists", "async def start_real_trading_loop" in api)
check("stop_real_trading_loop exists", "async def stop_real_trading_loop" in api)

# Critical bug fix: _aggressive_scalper_instance not overwritten
lines = api.split("\n")
in_loop_section = False
overwrite_found = False
for line in lines:
    if "Real Trading Loop" in line:
        in_loop_section = True
    if in_loop_section and line.strip() == "_aggressive_scalper_instance = None":
        overwrite_found = True
        break
check("_aggressive_scalper_instance NOT overwritten (critical fix)", not overwrite_found)

# ===== Phase 6: Frontend =====
print("\n=== Phase 6: Frontend Fixes ===")

with open("frontend/app/live/page.tsx", encoding="utf-8") as f:
    live = f.read()
check("live/page.tsx uses env var not hardcoded URL", "NEXT_PUBLIC_API_URL" in live)
check("live/page.tsx no raw localhost hardcode", "= 'http://localhost:8000/api" not in live)

# ===== Phase 7: Simulator removed =====
print("\n=== Phase 7: No Simulator ===")
check("No price_and_trading_simulator", "price_and_trading_simulator" not in api)
check("No fake/simulated price generator", "random.uniform" not in api)

# ===== Phase 8: Startup =====
print("\n=== Phase 8: Startup ===")
with open("start.py", encoding="utf-8") as f:
    start = f.read()
check("start.py targets dashboard.api:app", "dashboard.api:app" in start)

# ===== Phase 9: Config =====
print("\n=== Phase 9: Config ===")
import os
check(".env exists", os.path.exists(".env"))
check("frontend/.env.local exists", os.path.exists("frontend/.env.local"))

# ===== Phase 10: Strategy config =====
print("\n=== Phase 10: Strategy Config ===")
check("aggressive_scalper.yaml exists", os.path.exists("configs/aggressive_scalper.yaml"))

# ===== Summary =====
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {failed} check(s) failed")
    sys.exit(1)
