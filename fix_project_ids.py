#!/usr/bin/env python3
"""
Fix hardcoded project IDs in Python and SQL files
Replace with placeholder values for public repository
"""

import os
import re
from pathlib import Path

def fix_file(filepath, replacements):
    """Fix hardcoded values in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content
        for old_val, new_val in replacements.items():
            content = content.replace(old_val, new_val)

        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix all Python and SQL files."""

    # Define replacements
    replacements = {
        'gen-lang-client-0017660547': 'YOUR_PROJECT_ID',
        '"gen-lang-client-0017660547"': '"YOUR_PROJECT_ID"',
        "'gen-lang-client-0017660547'": "'YOUR_PROJECT_ID'",
        '670960352209': 'YOUR_PROJECT_NUMBER',
        'adafinoaiei.victor@gmail.com': 'your-email@example.com',
        'adafinoaiei.victor': 'your-username'
    }

    # Process Python files
    print("Processing Python files...")
    python_dir = Path('python_files')
    fixed_count = 0

    for py_file in python_dir.glob('*.py'):
        if fix_file(py_file, replacements):
            print(f"  ✅ Fixed: {py_file.name}")
            fixed_count += 1

    # Process SQL files
    print("\nProcessing SQL files...")
    sql_dir = Path('sql_files')

    for sql_file in sql_dir.glob('*.sql'):
        if fix_file(sql_file, replacements):
            print(f"  ✅ Fixed: {sql_file.name}")
            fixed_count += 1

    # Also check root directory Python files
    print("\nProcessing root Python files...")
    root_dir = Path('.')
    for py_file in root_dir.glob('*.py'):
        if py_file.name != 'fix_project_ids.py':  # Skip this script
            if fix_file(py_file, replacements):
                print(f"  ✅ Fixed: {py_file.name}")
                fixed_count += 1

    print(f"\n✅ Total files fixed: {fixed_count}")
    print("✅ All project IDs replaced with placeholders")
    print("\n⚠️  Users need to update these placeholders with their own values before running")

if __name__ == "__main__":
    main()