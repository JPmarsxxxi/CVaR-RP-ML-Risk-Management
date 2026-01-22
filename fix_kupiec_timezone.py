#!/usr/bin/env python3
"""
Fix timezone-aware vs timezone-naive timestamp mismatch in Kupiec Test.
The dates from cvar_rp_results have timezone info which causes the intersection to fail.
"""

import json
import sys

def fix_kupiec_timezone_issue(notebook_path):
    """Fix timezone handling in Kupiec Test date alignment."""

    print(f"Reading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    fixes_applied = 0

    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                source_lines = source
            else:
                source_lines = source.split('\n')
                source_lines = [line + '\n' if i < len(source_lines) - 1 else line
                               for i, line in enumerate(source_lines)]

            source_text = ''.join(source_lines) if isinstance(source_lines, list) else source_lines

            # Find the Kupiec Test cell with aligned_dates
            if 'PHASE 2.5: KUPIEC TEST' in source_text and 'aligned_dates = pd.DatetimeIndex(cvar_dates_list)' in source_text:
                print(f"\nFound Kupiec Test cell {cell_idx} - adding timezone fix")

                new_lines = []
                i = 0

                while i < len(source_lines):
                    line = source_lines[i]

                    # Find the aligned_dates creation and add timezone normalization
                    if 'aligned_dates = pd.DatetimeIndex(cvar_dates_list)' in line:
                        new_lines.append('# Convert dates to datetime index and normalize timezone\n')
                        new_lines.append('# CRITICAL: Remove timezone info to ensure proper date matching\n')
                        new_lines.append('aligned_dates = pd.DatetimeIndex(cvar_dates_list).tz_localize(None)\n')
                        i += 1
                        continue

                    # Also fix the returns_dict to use timezone-naive dates
                    if 'returns_dict = dict(zip(pd.to_datetime(cvar_rp_results' in line:
                        new_lines.append('# Build a dict for fast lookup of returns by date (remove timezone)\n')
                        new_lines.append('returns_dict = dict(zip(\n')
                        new_lines.append('    pd.to_datetime(cvar_rp_results[\'date\']).dt.tz_localize(None),\n')
                        new_lines.append('    cvar_rp_results[\'return\']\n')
                        new_lines.append('))\n')
                        i += 1
                        continue

                    new_lines.append(line)
                    i += 1

                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Added timezone normalization")

    if fixes_applied > 0:
        print(f"\n[OK] Applied {fixes_applied} fix(es)")
        print(f"Writing fixed notebook to: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook fixed successfully!")
        return True
    else:
        print("\n[WARNING] No fixes applied. Pattern may not have matched.")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    try:
        success = fix_kupiec_timezone_issue(notebook_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
