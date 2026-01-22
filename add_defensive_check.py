#!/usr/bin/env python3
"""
Add defensive date alignment check right before Kupiec Test call.
This ensures both series are aligned even if some dates failed during CVaR calculation.
"""

import json
import sys

def add_defensive_alignment_check(notebook_path):
    """Add common dates filtering right before kupiec_test call."""

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

            # Find the Kupiec Test cell
            if 'PHASE 2.5: KUPIEC TEST' in source_text and 'kupiec_results = kupiec_test(portfolio_returns, cvar_series' in source_text:
                print(f"\nFound Kupiec Test cell {cell_idx}")

                new_lines = []
                i = 0
                added_check = False

                while i < len(source_lines):
                    line = source_lines[i]

                    # Add defensive check right before the kupiec_test call
                    if 'kupiec_results = kupiec_test(portfolio_returns, cvar_series' in line and not added_check:
                        # Add the defensive alignment check
                        new_lines.append('# DEFENSIVE CHECK: Ensure both series have common dates\n')
                        new_lines.append('# This handles cases where some CVaR calculations may have failed\n')
                        new_lines.append('common_dates_check = cvar_series.index.intersection(portfolio_returns.index)\n')
                        new_lines.append('if len(common_dates_check) == 0:\n')
                        new_lines.append('    print(f"  ERROR: No common dates found!")\n')
                        new_lines.append('    print(f"  cvar_series index type: {type(cvar_series.index)}")\n')
                        new_lines.append('    print(f"  portfolio_returns index type: {type(portfolio_returns.index)}")\n')
                        new_lines.append('    print(f"  cvar_series has {len(cvar_series)} dates")\n')
                        new_lines.append('    print(f"  portfolio_returns has {len(portfolio_returns)} dates")\n')
                        new_lines.append('    if len(cvar_series) > 0:\n')
                        new_lines.append('        print(f"  cvar_series sample: {cvar_series.index[:3].tolist()}")\n')
                        new_lines.append('    if len(portfolio_returns) > 0:\n')
                        new_lines.append('        print(f"  portfolio_returns sample: {portfolio_returns.index[:3].tolist()}")\n')
                        new_lines.append('    raise ValueError("No common dates between returns and CVaR values - check date alignment")\n')
                        new_lines.append('elif len(common_dates_check) < len(cvar_series):\n')
                        new_lines.append('    print(f"  WARNING: Only {len(common_dates_check)}/{len(cvar_series)} dates align, filtering both series...")\n')
                        new_lines.append('    cvar_series = cvar_series.loc[common_dates_check]\n')
                        new_lines.append('    portfolio_returns = portfolio_returns.loc[common_dates_check]\n')
                        new_lines.append('    print(f"  After filtering: {len(cvar_series)} common dates")\n')
                        new_lines.append('\n')
                        added_check = True

                    new_lines.append(line)
                    i += 1

                if added_check:
                    cell['source'] = new_lines
                    fixes_applied += 1
                    print("  [OK] Added defensive alignment check before kupiec_test call")

    if fixes_applied > 0:
        print(f"\n[OK] Applied {fixes_applied} fix(es)")
        print(f"Writing fixed notebook to: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook fixed successfully!")
        return True
    else:
        print("\n[WARNING] No fixes applied.")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    try:
        success = add_defensive_alignment_check(notebook_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
