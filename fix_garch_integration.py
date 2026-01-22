#!/usr/bin/env python3
"""
Script to integrate GARCH volatilities into CVaR-RP optimization.
Fixes the critical issue where GARCH volatilities are calculated but never used.
"""

import json
import sys
import re

def fix_garch_integration(notebook_path):
    """Integrate GARCH volatilities into CVaR-RP optimization."""
    
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
            
            # Fix 1: Phase 2.1 - Store GARCH volatilities in a dictionary
            if 'PHASE 2.1: GARCH(1,1) VOLATILITY ESTIMATION' in source_text and 'garch_volatilities' not in source_text:
                print(f"\nFound Phase 2.1 in cell {cell_idx}")
                new_lines = []
                i = 0
                storing_vols = False
                while i < len(source_lines):
                    line = source_lines[i]
                    
                    # Find where volatilities are calculated but not stored
                    if '# Now estimate for all assets on training data' in line or 'Step 1.*Estimating GARCH' in line:
                        new_lines.append(line)
                        i += 1
                        # Add dictionary to store volatilities
                        new_lines.append('garch_volatilities_train = {}  # Store GARCH volatilities for each asset\n')
                        storing_vols = True
                        continue
                    
                    # Find the loop that calculates volatilities
                    if storing_vols and 'for asset in returns_train.columns:' in line:
                        new_lines.append(line)
                        i += 1
                        # Skip the print line
                        if i < len(source_lines) and 'Processing' in source_lines[i]:
                            i += 1
                        # Modify the volatility calculation to store it
                        while i < len(source_lines) and 'vol = estimate_garch_volatility' in source_lines[i]:
                            new_lines.append('    vol = estimate_garch_volatility(returns_train[asset])\n')
                            new_lines.append('    garch_volatilities_train[asset] = vol  # Store for use in optimization\n')
                            i += 1
                            # Skip the print line
                            if i < len(source_lines) and 'variance: {vol:.6f}' in source_lines[i]:
                                new_lines.append(source_lines[i])
                                i += 1
                            continue
                    
                    new_lines.append(line)
                    i += 1
                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Modified Phase 2.1 to store GARCH volatilities")
            
            # Fix 2: Phase 2.3 - Modify optimize_cvar_rp to accept and use GARCH volatilities
            if 'def optimize_cvar_rp(returns_data' in source_text and 'garch_volatilities' not in source_text:
                print(f"\nFound optimize_cvar_rp function in cell {cell_idx}")
                new_lines = []
                i = 0
                while i < len(source_lines):
                    line = source_lines[i]
                    
                    # Modify function signature
                    if 'def optimize_cvar_rp(returns_data, alpha=0.05' in line:
                        new_lines.append('def optimize_cvar_rp(returns_data, alpha=0.05, garch_volatilities=None, max_iter=3000, tol=1e-6, verbose=False):\n')
                        i += 1
                        # Update docstring
                        while i < len(source_lines) and ('\"\"\"' not in source_lines[i] or 'Parameters:' not in source_text[max(0, i-5):i+5]):
                            if 'Parameters:' in source_lines[i]:
                                new_lines.append(line)
                                i += 1
                                # Add garch_volatilities parameter
                                while i < len(source_lines) and '- returns_data:' in source_lines[i]:
                                    new_lines.append(source_lines[i])
                                    i += 1
                                new_lines.append('    - garch_volatilities: dict of GARCH-forecasted variances {asset: variance} (optional)\n')
                                new_lines.append('      If provided, builds covariance matrix from GARCH vols + correlations\n')
                                new_lines.append('      If None, uses historical sample covariance (default)\n')
                                continue
                            new_lines.append(line)
                            i += 1
                        continue
                    
                    # Replace covariance calculation
                    if '# Step 1: Calculate mean returns and covariance matrix' in line:
                        new_lines.append('# Step 1: Calculate mean returns and covariance matrix\n')
                        new_lines.append('    mu = returns_data.mean().values  # Mean returns (n,)\n')
                        new_lines.append('    \n')
                        new_lines.append('    # Build covariance matrix: use GARCH if provided, else sample covariance\n')
                        new_lines.append('    if garch_volatilities is not None:\n')
                        new_lines.append('        # Paper methodology: Build covariance from GARCH volatilities + correlations\n')
                        new_lines.append('        # Cov = diag(σ_garch) * Corr * diag(σ_garch)\n')
                        new_lines.append('        # This uses GARCH-forecasted volatilities instead of historical\n')
                        new_lines.append('        corr_matrix = returns_data.corr().values  # Correlation matrix\n')
                        new_lines.append('        # Get GARCH volatilities (variances) in same order as columns\n')
                        new_lines.append('        vol_vector = np.array([np.sqrt(garch_volatilities.get(asset, returns_data[asset].var())) \n')
                        new_lines.append('                                     for asset in returns_data.columns])\n')
                        new_lines.append('        # Build covariance: Cov_ij = σ_i * σ_j * Corr_ij\n')
                        new_lines.append('        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix\n')
                        new_lines.append('        if verbose:\n')
                        new_lines.append('            print(f\"  Using GARCH-forecasted volatilities for covariance matrix\")\n')
                        new_lines.append('            print(f\"  GARCH vol range: {vol_vector.min():.4f} to {vol_vector.max():.4f}\")\n')
                        new_lines.append('    else:\n')
                        new_lines.append('        # Fallback: use historical sample covariance\n')
                        new_lines.append('        cov_matrix = returns_data.cov().values  # Covariance matrix (n, n)\n')
                        new_lines.append('        if verbose:\n')
                        new_lines.append('            print(f\"  Using historical sample covariance matrix\")\n')
                        i += 1
                        # Skip old covariance line
                        while i < len(source_lines) and ('cov_matrix = returns_data.cov()' in source_lines[i] or 
                                                         'mu = returns_data.mean()' in source_lines[i]):
                            i += 1
                        continue
                    
                    new_lines.append(line)
                    i += 1
                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Modified optimize_cvar_rp to use GARCH volatilities")
            
            # Fix 3: Phase 2.4 - Modify backtest to calculate and pass GARCH volatilities
            if 'def backtest_cvar_rp(returns_data' in source_text and 'garch_volatilities' not in source_text:
                print(f"\nFound backtest_cvar_rp function in cell {cell_idx}")
                new_lines = []
                i = 0
                while i < len(source_lines):
                    line = source_lines[i]
                    
                    # Find where optimize_cvar_rp is called
                    if 'weights = optimize_cvar_rp(estimation_window' in line:
                        # Replace with GARCH-based call
                        new_lines.append('            # Step 1: Estimate GARCH volatilities for this window (Paper methodology)\n')
                        new_lines.append('            garch_vols = {}\n')
                        new_lines.append('            for asset in estimation_window.columns:\n')
                        new_lines.append('                try:\n')
                        new_lines.append('                    vol_var = estimate_garch_volatility(estimation_window[asset], window=len(estimation_window))\n')
                        new_lines.append('                    garch_vols[asset] = vol_var\n')
                        new_lines.append('                except Exception as e:\n')
                        new_lines.append('                    # Fallback to sample variance if GARCH fails\n')
                        new_lines.append('                    garch_vols[asset] = estimation_window[asset].var()\n')
                        new_lines.append('            \n')
                        new_lines.append('            # Step 2: Optimize weights using GARCH-forecasted volatilities\n')
                        new_lines.append('            weights = optimize_cvar_rp(estimation_window, alpha=alpha, garch_volatilities=garch_vols, verbose=False)\n')
                        i += 1
                        # Skip old optimize_cvar_rp line
                        continue
                    
                    new_lines.append(line)
                    i += 1
                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Modified backtest_cvar_rp to calculate and use GARCH volatilities")
            
            # Fix 4: Phase 4.1 - Modify ML-enhanced backtest to use GARCH
            if 'def backtest_ml_enhanced_cvar_rp(returns_data' in source_text and 'garch_volatilities' not in source_text:
                print(f"\nFound backtest_ml_enhanced_cvar_rp function in cell {cell_idx}")
                new_lines = []
                i = 0
                while i < len(source_lines):
                    line = source_lines[i]
                    
                    # Find where optimize_cvar_rp is called
                    if 'cvar_weights = optimize_cvar_rp(estimation_window' in line:
                        # Replace with GARCH-based call
                        new_lines.append('            # Step 1: Estimate GARCH volatilities for this window (Paper methodology)\n')
                        new_lines.append('            garch_vols = {}\n')
                        new_lines.append('            for asset in estimation_window.columns:\n')
                        new_lines.append('                try:\n')
                        new_lines.append('                    vol_var = estimate_garch_volatility(estimation_window[asset], window=len(estimation_window))\n')
                        new_lines.append('                    garch_vols[asset] = vol_var\n')
                        new_lines.append('                except Exception as e:\n')
                        new_lines.append('                    # Fallback to sample variance if GARCH fails\n')
                        new_lines.append('                    garch_vols[asset] = estimation_window[asset].var()\n')
                        new_lines.append('            \n')
                        new_lines.append('            # Step 2: Calculate baseline CVaR-RP weights using GARCH volatilities\n')
                        new_lines.append('            cvar_weights = optimize_cvar_rp(estimation_window, alpha=alpha, garch_volatilities=garch_vols, verbose=False)\n')
                        i += 1
                        # Skip old optimize_cvar_rp line
                        continue
                    
                    new_lines.append(line)
                    i += 1
                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Modified backtest_ml_enhanced_cvar_rp to use GARCH volatilities")
    
    if fixes_applied > 0:
        print(f"\n[OK] Applied {fixes_applied} fix(es)")
        print(f"Writing fixed notebook to: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook fixed successfully!")
        return True
    else:
        print("\n[WARNING] No fixes applied. The GARCH integration might already be correct or the pattern didn't match.")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    
    try:
        fix_garch_integration(notebook_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
