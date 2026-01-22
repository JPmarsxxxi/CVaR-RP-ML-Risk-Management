#!/usr/bin/env python3
"""
Script to fix the Kupiec Test date alignment issue.
The problem is the filtering logic that tries to align portfolio returns with CVaR dates.
"""

import json
import sys

def fix_kupiec_date_alignment(notebook_path):
    """Fix the date filtering logic in Kupiec Test."""

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

            # Find the Kupiec Test cell with the problematic filtering
            if 'PHASE 2.5: KUPIEC TEST' in source_text and 'cvar_dates_set = set(pd.to_datetime(cvar_dates_list))' in source_text:
                print(f"\nFound Kupiec Test cell {cell_idx} with date filtering issue")

                new_lines = []
                i = 0

                while i < len(source_lines):
                    line = source_lines[i]

                    # Find the CVaR Series creation and replace everything up to Kupiec test call
                    if '# Create CVaR Series' in line or 'cvar_series = pd.Series(cvar_values_list' in line:
                        # Add fixed alignment logic
                        new_lines.append('# Create aligned series - use same index for both\n')
                        new_lines.append('# Convert dates to datetime index once\n')
                        new_lines.append('aligned_dates = pd.DatetimeIndex(cvar_dates_list)\n')
                        new_lines.append('\n')
                        new_lines.append('# Create CVaR series\n')
                        new_lines.append('cvar_series = pd.Series(cvar_values_list, index=aligned_dates, name=\'CVaR\')\n')
                        new_lines.append('\n')
                        new_lines.append('# Create portfolio returns series with the same index\n')
                        new_lines.append('# Build a dict for fast lookup of returns by date\n')
                        new_lines.append('returns_dict = dict(zip(pd.to_datetime(cvar_rp_results[\'date\']), cvar_rp_results[\'return\']))\n')
                        new_lines.append('\n')
                        new_lines.append('# Get returns for the aligned dates\n')
                        new_lines.append('aligned_returns = [returns_dict.get(dt, np.nan) for dt in aligned_dates]\n')
                        new_lines.append('portfolio_returns = pd.Series(aligned_returns, index=aligned_dates, name=\'Portfolio_Return\')\n')
                        new_lines.append('\n')
                        new_lines.append('# Verify no NaN values (all dates should match)\n')
                        new_lines.append('if portfolio_returns.isna().any():\n')
                        new_lines.append('    print(f\"  WARNING: {portfolio_returns.isna().sum()} missing returns - dates may not align\")\n')
                        new_lines.append('    # Drop NaN values\n')
                        new_lines.append('    valid_idx = ~portfolio_returns.isna()\n')
                        new_lines.append('    portfolio_returns = portfolio_returns[valid_idx]\n')
                        new_lines.append('    cvar_series = cvar_series[valid_idx]\n')
                        new_lines.append('\n')
                        new_lines.append('print(f\"  Calculated CVaR for {len(cvar_series)} periods\")\n')
                        new_lines.append('print(f\"  Portfolio returns available for {len(portfolio_returns)} periods\")\n')
                        new_lines.append('print(f\"  Date alignment check: {len(cvar_series.index.intersection(portfolio_returns.index))} common dates\")\n')
                        new_lines.append('\n')

                        # Skip all the old code until we find the Kupiec test call
                        i += 1
                        while i < len(source_lines) and 'Run Kupiec Test' not in source_lines[i]:
                            i += 1
                        continue

                    new_lines.append(line)
                    i += 1

                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Fixed date alignment logic")

    if fixes_applied > 0:
        print(f"\n[OK] Applied {fixes_applied} fix(es)")
        print(f"Writing fixed notebook to: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook fixed successfully!")
        return True
    else:
        print("\n[WARNING] No fixes applied. Pattern may not have matched.")
        print("Looking for alternative patterns...")

        # Try alternative approach - look for the specific problematic line
        for cell_idx, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = cell['source']
                source_text = ''.join(source) if isinstance(source, list) else source

                if 'pd.to_datetime(cvar_rp_results[\'date\'].values).isin(cvar_dates_set)' in source_text:
                    print(f"\nFound alternative pattern in cell {cell_idx}")

                    # Apply fix to this specific pattern
                    if isinstance(source, list):
                        source_lines = source
                    else:
                        source_lines = source.split('\n')
                        source_lines = [line + '\n' if i < len(source_lines) - 1 else line
                                       for i, line in enumerate(source_lines)]

                    new_lines = []
                    i = 0

                    while i < len(source_lines):
                        line = source_lines[i]

                        if 'cvar_series = pd.Series(cvar_values_list' in line:
                            # Replace with fixed version
                            new_lines.append('# Create aligned series - FIXED\n')
                            new_lines.append('aligned_dates = pd.DatetimeIndex(cvar_dates_list)\n')
                            new_lines.append('cvar_series = pd.Series(cvar_values_list, index=aligned_dates, name=\'CVaR\')\n')
                            new_lines.append('\n')
                            new_lines.append('# Create matching portfolio returns series\n')
                            new_lines.append('returns_dict = dict(zip(pd.to_datetime(cvar_rp_results[\'date\']), cvar_rp_results[\'return\']))\n')
                            new_lines.append('aligned_returns = [returns_dict[dt] for dt in aligned_dates]\n')
                            new_lines.append('portfolio_returns = pd.Series(aligned_returns, index=aligned_dates, name=\'Portfolio_Return\')\n')
                            new_lines.append('\n')

                            # Skip old code
                            i += 1
                            while i < len(source_lines) and ('portfolio_returns' in source_lines[i] or 'cvar_dates_set' in source_lines[i] or 'portfolio_returns_filtered' in source_lines[i]):
                                i += 1
                            continue

                        new_lines.append(line)
                        i += 1

                    cell['source'] = new_lines
                    fixes_applied += 1
                    print("  [OK] Applied alternative fix")

                    print(f"\n[OK] Applied {fixes_applied} fix(es)")
                    print(f"Writing fixed notebook to: {notebook_path}")
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        json.dump(notebook, f, indent=1, ensure_ascii=False)
                    print("[OK] Notebook fixed successfully!")
                    return True

        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    try:
        success = fix_kupiec_date_alignment(notebook_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
