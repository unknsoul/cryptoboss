"""Cleanup dead code from routes.py"""
import os

path = os.path.join(os.path.dirname(__file__), "src", "api", "routes.py")
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Keep only lines up to and including the first if __name__ block (line 1647)
# Find the first "if __name__" line
cutoff = None
for i, line in enumerate(lines):
    if 'if __name__ == "__main__":' in line or "if __name__ == '__main__':" in line:
        # Keep this line + next 2 lines (import uvicorn, uvicorn.run)
        cutoff = i + 3  # include blank line after
        break

if cutoff:
    kept = lines[:cutoff]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(kept)
    print(f"Cleaned routes.py: kept {cutoff} of {len(lines)} lines")
else:
    print("Could not find __main__ block")
