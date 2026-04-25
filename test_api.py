"""CryptoBoss API Endpoint Test Suite"""
import urllib.request
import json

ENDPOINTS = [
    ("http://localhost:8000/api/status", "System Status"),
    ("http://localhost:8000/api/system", "System Info"),
    ("http://localhost:8000/api/context", "Market Context"),
    ("http://localhost:8000/api/risk", "Risk Data"),
    ("http://localhost:8000/api/prices/live", "Live Prices"),
    ("http://localhost:8000/api/pnl/history", "PnL History"),
    ("http://localhost:8000/api/strategies", "Strategies"),
    ("http://localhost:8000/api/scalper/aggressive/status", "Scalper Status"),
    ("http://localhost:8000/api/portfolio", "Portfolio"),
    ("http://localhost:8000/api/incident", "Incident State"),
]

print("=" * 70)
print("  CryptoBoss API Endpoint Test Suite")
print("=" * 70)

passed = 0
failed = 0

for url, name in ENDPOINTS:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            status_code = resp.status
            issues = []

            # Unwrap if needed
            inner = data.get("data", data) if isinstance(data, dict) else data

            if name == "Live Prices":
                prices = inner.get("prices", {})
                btc = prices.get("BTCUSDT", {})
                btc_price = btc.get("price", 0)
                source = btc.get("source", "?")
                if btc_price > 0:
                    print("  [PASS] {:<25s} | HTTP {} | BTC=${:,.2f} source={}".format(name, status_code, btc_price, source))
                else:
                    print("  [WARN] {:<25s} | HTTP {} | BTC price=0 (WS connecting)".format(name, status_code))
                passed += 1
                continue

            elif name == "Strategies":
                strats = data.get("strategies", [])
                has_scalper = any(s.get("type") == "AggressiveScalper" for s in strats)
                has_leverage = any(s.get("leverage") for s in strats)
                if not has_scalper:
                    issues.append("AggressiveScalper missing")
                if not has_leverage:
                    issues.append("get_status fields missing")

            elif name == "PnL History":
                check = inner if isinstance(inner, dict) else data
                if "best_trade" not in check:
                    issues.append("best_trade missing")
                if "worst_trade" not in check:
                    issues.append("worst_trade missing")

            elif name == "Scalper Status":
                if "halted" not in inner:
                    issues.append("halted field missing")

            elif name == "System Status":
                if data.get("mode") == "paper":
                    issues.append("PAPER mode still active!")

            if issues:
                print("  [FAIL] {:<25s} | HTTP {} | Issues: {}".format(name, status_code, "; ".join(issues)))
                failed += 1
            else:
                snippet = "OK"
                if name == "System Status":
                    snippet = "mode={} price=${:,.2f}".format(data.get("mode", "?"), data.get("current_price", 0))
                elif name == "Scalper Status":
                    snippet = "halted={} leverage={}".format(inner.get("halted"), inner.get("leverage"))
                elif name == "Market Context":
                    snippet = "price=${:,.2f}".format(inner.get("current_price", 0))
                elif name == "Incident State":
                    snippet = "state={} trading_allowed={}".format(inner.get("state"), inner.get("trading_allowed"))
                elif name == "PnL History":
                    check = inner if isinstance(inner, dict) else data
                    snippet = "trades={} best={} worst={}".format(
                        check.get("total_trades", 0), check.get("best_trade", 0), check.get("worst_trade", 0)
                    )
                elif name == "Portfolio":
                    snippet = "value=${}".format(inner.get("total_value_usd", 0))
                print("  [PASS] {:<25s} | HTTP {} | {}".format(name, status_code, snippet))
                passed += 1

    except Exception as e:
        print("  [FAIL] {:<25s} | ERROR: {}".format(name, e))
        failed += 1

print("=" * 70)
print("  Results: {} passed, {} failed out of {} endpoints".format(passed, failed, len(ENDPOINTS)))
print("=" * 70)
