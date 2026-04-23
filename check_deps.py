"""Quick dependency checker and installer."""
import subprocess
import sys

def check_and_install():
    missing = []
    deps = {
        'dotenv': 'python-dotenv',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'aiohttp': 'aiohttp',
        'ccxt': 'ccxt',
    }
    
    for module, pip_name in deps.items():
        try:
            __import__(module)
            print(f"  OK: {module}")
        except ImportError:
            print(f"  MISSING: {module} -> will install {pip_name}")
            missing.append(pip_name)
    
    if missing:
        print(f"\nInstalling {len(missing)} missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("Installation complete!")
    else:
        print("\nAll dependencies are installed!")

if __name__ == "__main__":
    check_and_install()
