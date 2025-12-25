import json
from datetime import datetime

trades = json.load(open('data/trades.json'))

print(f"\n=== LAST 5 TRADES ANALYSIS ===\n")

for i, t in enumerate(trades[-5:]):
    entry = t.get('entry_price', 0)
    tp = t.get('take_profit', 0)
    sl = t.get('stop_loss', 0)
    side = t.get('side', '?')
    
    if side == 'LONG':
        tp_distance = tp - entry if tp else 0
        sl_distance = entry - sl if sl else 0
    else:
        tp_distance = entry - tp if tp else 0
        sl_distance = sl - entry if sl else 0
    
    rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
    
    print(f"Trade {len(trades)-4+i}:")
    print(f"  Side: {side}")
    print(f"  Entry: ${entry:,.2f}")
    print(f"  TP: ${tp:,.2f} (distance: ${tp_distance:,.2f})")
    print(f"  SL: ${sl:,.2f} (distance: ${sl_distance:,.2f})")
    print(f"  R:R Ratio: {rr_ratio:.2f}:1")
    print(f"  Return: {t.get('return_pct', 0):.2f}%")
    print(f"  Time: {t.get('timestamp', '')[:19]}")
    print()

# Calculate time between trades
print("\n=== TRADE FREQUENCY ===\n")
for i in range(max(0, len(trades)-5), len(trades)-1):
    t1 = trades[i]
    t2 = trades[i+1]
    ts1 = t1.get('timestamp', '')[:19]
    ts2 = t2.get('timestamp', '')[:19]
    try:
        dt1 = datetime.fromisoformat(ts1)
        dt2 = datetime.fromisoformat(ts2)
        diff = (dt2 - dt1).total_seconds() / 60
        print(f"Trade {i+1} to {i+2}: {diff:.1f} minutes apart")
    except:
        pass

print(f"\nTotal trades: {len(trades)}")
