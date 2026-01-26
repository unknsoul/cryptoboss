#!/usr/bin/env python3
"""
Project Cleanup Script
Removes unnecessary files, organizes structure, and prepares for production.

Run with: python cleanup_project.py --dry-run  # Preview changes
          python cleanup_project.py             # Execute cleanup
"""

import os
import shutil
from pathlib import Path
import argparse

# Files to DELETE (debug outputs, logs, temp files)
FILES_TO_DELETE = [
    # Debug output files
    "analysis_output.txt",
    "console_output.txt",
    "debug_output.txt",
    "debug_trace.txt",
    "error_log.txt",
    "latest_run.txt",
    "loc_stats.txt",
    "my_debug.txt",
    "trade_analysis.txt",
    
    # Debug scripts
    "debug_bot_execution.py",
    "debug_imports.py",
    "debug_init.py",
    "debug_signals.py",
    
    # Temporary test files (keep formal tests in tests/)
    "test_accuracy.py",
    "test_dca_quick.py",
    "test_feature_engineering.py",
    "test_final_components.py",
    "test_ml_pipeline.py",
    "test_real_backtest.py",
    "test_risk_engine.py",
    "test_signal_logic.py",
    "test_tier1_improvements.py",
    
    # Obsolete files
    "FIX_RISK_REWARD.py",
    "INTEGRATION_GUIDE.py",
    "analyze_trades.py",
    "count_loc.py",
    "encrypt_config.py",
    "install.py",
    
    # Image outputs (regenerate as needed)
    "backtest_results.png",
    "tier1_comparison.png",
]

# Directories to DELETE (if empty or obsolete)
DIRS_TO_DELETE = [
    "__pycache__",
    "test_backups",
    "integration",  # Empty or duplicate
]

# Files to KEEP (important)
FILES_TO_KEEP = [
    "README.md",
    "requirements.txt",
    "pytest.ini",
    "Dockerfile",
    "docker-compose.yml",
    ".env",
    ".gitignore",
    "LICENSE",
    "BINANCE_CONFIG_GUIDE.md",
    "run_trading_bot.py",
    "run_backtest.py",
    "download_data.py",
    "setup_binance.py",
    "train_ml.py",
]


def cleanup_pycache(base_path: Path, dry_run: bool):
    """Remove all __pycache__ directories."""
    count = 0
    for pycache in base_path.rglob("__pycache__"):
        if pycache.is_dir():
            if dry_run:
                print(f"  Would delete: {pycache}")
            else:
                shutil.rmtree(pycache)
                print(f"  Deleted: {pycache}")
            count += 1
    return count


def cleanup_files(base_path: Path, dry_run: bool):
    """Remove specified files."""
    count = 0
    for filename in FILES_TO_DELETE:
        filepath = base_path / filename
        if filepath.exists():
            if dry_run:
                print(f"  Would delete: {filepath}")
            else:
                filepath.unlink()
                print(f"  Deleted: {filepath}")
            count += 1
    return count


def cleanup_empty_dirs(base_path: Path, dry_run: bool):
    """Remove empty directories."""
    count = 0
    for dirpath in sorted(base_path.rglob("*"), reverse=True):
        if dirpath.is_dir():
            try:
                # Check if directory is empty (excluding hidden files)
                contents = list(dirpath.iterdir())
                visible_contents = [c for c in contents if not c.name.startswith('.')]
                
                if not visible_contents or dirpath.name in DIRS_TO_DELETE:
                    if dry_run:
                        print(f"  Would delete empty dir: {dirpath}")
                    else:
                        shutil.rmtree(dirpath)
                        print(f"  Deleted empty dir: {dirpath}")
                    count += 1
            except PermissionError:
                pass
    return count


def show_final_structure(base_path: Path):
    """Show the cleaned project structure."""
    print("\n" + "=" * 60)
    print("FINAL PROJECT STRUCTURE")
    print("=" * 60)
    
    important_dirs = ["src", "core", "configs", "tests", "data"]
    
    for dir_name in important_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"\n📁 {dir_name}/")
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    child_count = len(list(item.glob("*.py")))
                    print(f"  📂 {item.name}/ ({child_count} files)")
                elif item.suffix == ".py":
                    print(f"  📄 {item.name}")


def main():
    parser = argparse.ArgumentParser(description="Clean up project structure")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    args = parser.parse_args()
    
    base_path = Path(__file__).parent
    
    print("=" * 60)
    print("CRYPTOBOSS PROJECT CLEANUP")
    print("=" * 60)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made\n")
    else:
        print("\n⚠️  EXECUTING CLEANUP - Files will be deleted!\n")
    
    # Cleanup steps
    print("\n1. Removing __pycache__ directories...")
    pycache_count = cleanup_pycache(base_path, args.dry_run)
    print(f"   → {pycache_count} directories")
    
    print("\n2. Removing debug/temp files...")
    files_count = cleanup_files(base_path, args.dry_run)
    print(f"   → {files_count} files")
    
    print("\n3. Removing empty directories...")
    dirs_count = cleanup_empty_dirs(base_path, args.dry_run)
    print(f"   → {dirs_count} directories")
    
    # Summary
    total = pycache_count + files_count + dirs_count
    print("\n" + "=" * 60)
    
    if args.dry_run:
        print(f"CLEANUP PREVIEW: Would remove {total} items")
        print("Run without --dry-run to execute cleanup")
    else:
        print(f"CLEANUP COMPLETE: Removed {total} items")
        show_final_structure(base_path)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
