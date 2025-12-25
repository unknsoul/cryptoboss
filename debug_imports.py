import sys
import os
sys.path.insert(0, '.')

try:
    print("Importing settings...")
    from core.config import settings
    print(f"Settings loaded: {settings.INITIAL_CAPITAL}")

    print("Importing SQLiteManager...")
    from core.storage.database import SQLiteManager
    db = SQLiteManager()
    print("Database initialized.")

    print("Importing Dashboard App...")
    import dashboard.app
    print("Dashboard App imported.")
    
    print("Success!")

except Exception as e:
    import traceback
    traceback.print_exc()
