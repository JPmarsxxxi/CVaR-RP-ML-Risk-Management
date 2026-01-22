#!/usr/bin/env python3
"""
Script to fix the Kupiec Test date alignment issue in the notebook.
Fixes the ValueError: No common dates between returns and CVaR values.
"""

import json
import sys

def fix_kupiec_test(notebook_path):
    """Fix Kupiec Test to align CVaR dates with portfolio return dates."""

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
            if 'PHASE 2.5: KUPIEC TEST' in source_text and 'ValueError: No common dates' not in source_text:
                print(f"\nFound Kupiec Test cell {cell_idx}")

                # Check if this is the cell that needs fixing
                if 'for i, month_end in enumerate(monthly_returns_backtest.index):' in source_text:
                    print("  Found the problematic loop - applying fix...")

                    new_lines = []
                    i = 0
                    skip_mode = False

                    while i < len(source_lines):
                        line = source_lines[i]

                        # Replace the CVaR calculation section
                        if '[Step 1] Calculating CVaR values for each backtest period' in line:
                            # Keep the print statement
                            new_lines.append(line)
                            i += 1
                            # Keep the note
                            if i < len(source_lines) and 'Note: We need to recalculate CVaR' in source_lines[i]:
                                new_lines.append(source_lines[i])
                                i += 1

                            # Skip blank lines
                            while i < len(source_lines) and source_lines[i].strip() == '\n':
                                i += 1

                            # Add the fixed calculation
                            new_lines.append('\n')
                            new_lines.append('# FIXED: Iterate over actual backtest results to ensure date alignment\n')
                            new_lines.append('cvar_values_list = []\n')
                            new_lines.append('cvar_dates_list = []\n')
                            new_lines.append('\n')
                            new_lines.append('# Get the backtest data\n')
                            new_lines.append('backtest_data = pd.concat([returns_val, returns_test])\n')
                            new_lines.append('\n')
                            new_lines.append('# Iterate over each backtest result (ensures date alignment)\n')
                            new_lines.append('for idx, row in cvar_rp_results.iterrows():\n')
                            new_lines.append('    month_end = row[\'date\']\n')
                            new_lines.append('    weights = row[\'weights\']\n')
                            new_lines.append('    \n')
                            new_lines.append('    # Get estimation window (same as backtest)\n')
                            new_lines.append('    daily_data_up_to_month = backtest_data[backtest_data.index < month_end]\n')
                            new_lines.append('    \n')
                            new_lines.append('    min_days = 20\n')
                            new_lines.append('    if len(daily_data_up_to_month) < min_days:\n')
                            new_lines.append('        continue\n')
                            new_lines.append('    \n')
                            new_lines.append('    # Calculate lookback date: 3 calendar months before month_end\n')
                            new_lines.append('    from dateutil.relativedelta import relativedelta\n')
                            new_lines.append('    lookback_date = month_end - relativedelta(months=3)\n')
                            new_lines.append('    \n')
                            new_lines.append('    # Use all data from lookback_date to month_end (exclusive)\n')
                            new_lines.append('    estimation_window = daily_data_up_to_month[daily_data_up_to_month.index >= lookback_date]\n')
                            new_lines.append('    \n')
                            new_lines.append('    if len(estimation_window) < min_days:\n')
                            new_lines.append('        continue\n')
                            new_lines.append('    \n')
                            new_lines.append('    try:\n')
                            new_lines.append('        # Calculate GARCH volatilities (same as backtest)\n')
                            new_lines.append('        garch_vols = {}\n')
                            new_lines.append('        for asset in estimation_window.columns:\n')
                            new_lines.append('            try:\n')
                            new_lines.append('                vol_var = estimate_garch_volatility(estimation_window[asset], window=len(estimation_window))\n')
                            new_lines.append('                garch_vols[asset] = vol_var\n')
                            new_lines.append('            except Exception as e:\n')
                            new_lines.append('                garch_vols[asset] = estimation_window[asset].var()\n')
                            new_lines.append('        \n')
                            new_lines.append('        # Calculate portfolio-level CVaR for this period\n')
                            new_lines.append('        mu = estimation_window.mean().values\n')
                            new_lines.append('        corr_matrix = estimation_window.corr().values\n')
                            new_lines.append('        \n')
                            new_lines.append('        # Build covariance matrix from GARCH volatilities\n')
                            new_lines.append('        vol_vector = np.array([np.sqrt(garch_vols.get(asset, estimation_window[asset].var())) \n')
                            new_lines.append('                              for asset in estimation_window.columns])\n')
                            new_lines.append('        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix\n')
                            new_lines.append('        \n')
                            new_lines.append('        # Calculate portfolio CVaR\n')
                            new_lines.append('        portfolio_cvar = calculate_portfolio_cvar(weights, mu, cov_matrix, alpha=0.05)\n')
                            new_lines.append('        \n')
                            new_lines.append('        cvar_values_list.append(portfolio_cvar)\n')
                            new_lines.append('        cvar_dates_list.append(month_end)\n')
                            new_lines.append('        \n')
                            new_lines.append('    except Exception as e:\n')
                            new_lines.append('        print(f"  Warning: Error calculating CVaR for {month_end.date()}: {str(e)}")\n')
                            new_lines.append('        continue\n')
                            new_lines.append('\n')
                            new_lines.append('# Create aligned series using the same dates\n')
                            new_lines.append('cvar_series = pd.Series(cvar_values_list, index=pd.DatetimeIndex(cvar_dates_list), name=\'CVaR\')\n')
                            new_lines.append('portfolio_returns = pd.Series(\n')
                            new_lines.append('    [cvar_rp_results[cvar_rp_results[\'date\'] == dt][\'return\'].values[0] for dt in cvar_dates_list],\n')
                            new_lines.append('    index=pd.DatetimeIndex(cvar_dates_list),\n')
                            new_lines.append('    name=\'Portfolio_Return\'\n')
                            new_lines.append(')\n')
                            new_lines.append('\n')
                            new_lines.append('print(f"  Calculated CVaR for {len(cvar_series)} periods")\n')
                            new_lines.append('print(f"  Portfolio returns available for {len(portfolio_returns)} periods")\n')
                            new_lines.append('print(f"  Date alignment check: {len(cvar_series.index.intersection(portfolio_returns.index))} common dates")\n')
                            new_lines.append('\n')

                            # Skip all old lines until we find the "Run Kupiec Test" section
                            skip_mode = True
                            while i < len(source_lines) and '[Step 2] Running Kupiec Test' not in source_lines[i]:
                                i += 1
                            skip_mode = False
                            continue

                        if not skip_mode:
                            new_lines.append(line)
                        i += 1

                    cell['source'] = new_lines
                    fixes_applied += 1
                    print("  [OK] Fixed Kupiec Test date alignment")

    if fixes_applied > 0:
        print(f"\n[OK] Applied {fixes_applied} fix(es)")
        print(f"Writing fixed notebook to: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook fixed successfully!")
        return True
    else:
        print("\n[WARNING] No fixes applied. The Kupiec test might already be fixed or the pattern didn't match.")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    try:
        success = fix_kupiec_test(notebook_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
