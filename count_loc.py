import os
from pathlib import Path

def count_loc(file_path):
    """Count lines of code excluding comments and blank lines"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        loc = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip blank lines
            if not stripped:
                continue
            
            # Handle multiline comments/docstrings
            if '"""' in stripped or "'''" in stripped:
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    continue
                in_multiline_comment = not in_multiline_comment
                continue
            
            if in_multiline_comment:
                continue
            
            # Skip single line comments
            if stripped.startswith('#'):
                continue
            
            # Count as code line
            loc += 1
        
        return loc
    except Exception as e:
        return 0

# Count all Python files
total_loc = 0
file_count = 0
core_loc = 0
dashboard_loc = 0
root_loc = 0
files_by_category = {}

for root, dirs, files in os.walk('.'):
    # Skip __pycache__, .git, etc
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'test_backups']
    
    for file in files:
        if file.endswith('.py'):
            file_path = Path(root) / file
            loc = count_loc(file_path)
            total_loc += loc
            file_count += 1
            
            # Categorize
            path_str = str(file_path)
            if 'core' in path_str:
                core_loc += loc
                category = path_str.split(os.sep)[1] if len(path_str.split(os.sep)) > 1 else 'core'
                files_by_category[category] = files_by_category.get(category, 0) + loc
            elif 'dashboard' in path_str:
                dashboard_loc += loc
            else:
                root_loc += loc

print(f"=" * 60)
print(f"PROJECT CODE STATISTICS (Excluding Comments & Blank Lines)")
print(f"=" * 60)
print(f"")
print(f"Total Python Files: {file_count}")
print(f"Total Lines of Code: {total_loc:,}")
print(f"")
print(f"BREAKDOWN BY CATEGORY:")
print(f"-" * 60)
print(f"  Core Modules:      {core_loc:>8,} lines ({core_loc/total_loc*100:.1f}%)")
print(f"  Dashboard:         {dashboard_loc:>8,} lines ({dashboard_loc/total_loc*100:.1f}%)")
print(f"  Root Scripts:      {root_loc:>8,} lines ({root_loc/total_loc*100:.1f}%)")
print(f"")
print(f"TOP MODULES:")
print(f"-" * 60)
sorted_modules = sorted(files_by_category.items(), key=lambda x: x[1], reverse=True)[:10]
for module, lines in sorted_modules:
    print(f"  {module:<25} {lines:>8,} lines")
print(f"=" * 60)
