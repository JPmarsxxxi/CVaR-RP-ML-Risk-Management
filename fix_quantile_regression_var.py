#!/usr/bin/env python3
"""
Script to add quantile regression for VaR calculation as per paper (Section 4.3.1, Page 7).
Replaces parametric normal distribution approach with quantile regression.
"""

import json
import sys

def fix_quantile_regression_var(notebook_path):
    """Add quantile regression for VaR calculation."""
    
    print(f"Reading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixes_applied = 0
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source
            
            # Find the calculate_portfolio_cvar function
            if 'def calculate_portfolio_cvar' in source_text:
                print(f"\n[Cell {cell_idx}] Adding quantile regression for VaR calculation...")
                
                # Read the current function
                if isinstance(source, list):
                    source_lines = source
                else:
                    source_lines = source.split('\n')
                    source_lines = [line + '\n' if i < len(source_lines) - 1 else line 
                                   for i, line in enumerate(source_lines)]
                
                new_lines = []
                i = 0
                modified = False
                
                while i < len(source_lines):
                    line = source_lines[i]
                    
                    # Add import for quantile regression at the top of the cell
                    if i == 0 and 'from statsmodels.regression.quantile_regression import QuantReg' not in source_text:
                        new_lines.append("from statsmodels.regression.quantile_regression import QuantReg\n")
                        new_lines.append("\n")
                        modified = True
                    
                    # Replace the calculate_portfolio_cvar function
                    if 'def calculate_portfolio_cvar(weights, mu, cov_matrix, alpha=0.05):' in line:
                        # Add new function with quantile regression option
                        new_lines.append("def calculate_var_quantile_regression(returns, alpha=0.05):\n")
                        new_lines.append("    \"\"\"\n")
                        new_lines.append("    Calculate VaR using quantile regression (Paper Section 4.3.1, Page 7).\n")
                        new_lines.append("    \n")
                        new_lines.append("    Paper: \"Following the estimation of asset price volatility using the GARCH(1,1) model,\n")
                        new_lines.append("    the quantile regression method is applied to calculate the VaR values.\"\n")
                        new_lines.append("    \n")
                        new_lines.append("    Parameters:\n")
                        new_lines.append("    - returns: numpy array or pandas Series of returns\n")
                        new_lines.append("    - alpha: tail probability (default 0.05 = 5% tail)\n")
                        new_lines.append("    \n")
                        new_lines.append("    Returns:\n")
                        new_lines.append("    - VaR value (scalar)\n")
                        new_lines.append("    \"\"\"\n")
                        new_lines.append("    # Convert to numpy array if needed\n")
                        new_lines.append("    if hasattr(returns, 'values'):\n")
                        new_lines.append("        y = returns.values.flatten()\n")
                        new_lines.append("    else:\n")
                        new_lines.append("        y = np.array(returns).flatten()\n")
                        new_lines.append("    \n")
                        new_lines.append("    # Remove NaN values\n")
                        new_lines.append("    y = y[~np.isnan(y)]\n")
                        new_lines.append("    \n")
                        new_lines.append("    if len(y) < 10:\n")
                        new_lines.append("        # Fallback to empirical quantile if too few observations\n")
                        new_lines.append("        return np.quantile(y, alpha)\n")
                        new_lines.append("    \n")
                        new_lines.append("    # Quantile regression with intercept only (marginal distribution)\n")
                        new_lines.append("    X = np.ones((len(y), 1))  # Intercept only\n")
                        new_lines.append("    \n")
                        new_lines.append("    try:\n")
                        new_lines.append("        model = QuantReg(y, X)\n")
                        new_lines.append("        result = model.fit(q=alpha)\n")
                        new_lines.append("        var = result.params[0]  # Intercept parameter\n")
                        new_lines.append("        return var\n")
                        new_lines.append("    except Exception as e:\n")
                        new_lines.append("        # Fallback to empirical quantile if quantile regression fails\n")
                        new_lines.append("        return np.quantile(y, alpha)\n")
                        new_lines.append("\n")
                        new_lines.append("\n")
                        new_lines.append("def calculate_portfolio_cvar(weights, mu, cov_matrix, alpha=0.05, returns_data=None, use_quantile_regression=True):\n")
                        new_lines.append("    \"\"\"\n")
                        new_lines.append("    Calculate portfolio-level Conditional Value at Risk (CVaR).\n")
                        new_lines.append("    \n")
                        new_lines.append("    Paper methodology (Section 4.3.1, Page 7):\n")
                        new_lines.append("    1. Estimate GARCH(1,1) volatility (already done)\n")
                        new_lines.append("    2. Use quantile regression to calculate VaR\n")
                        new_lines.append("    3. Calculate CVaR from VaR\n")
                        new_lines.append("    \n")
                        new_lines.append("    This function supports both:\n")
                        new_lines.append("    - Quantile regression approach (Paper method, default)\n")
                        new_lines.append("    - Parametric normal approach (fallback)\n")
                        new_lines.append("    \n")
                        new_lines.append("    Parameters:\n")
                        new_lines.append("    - weights: numpy array of portfolio weights (must sum to 1)\n")
                        new_lines.append("    - mu: numpy array of mean returns for each asset\n")
                        new_lines.append("    - cov_matrix: numpy array of covariance matrix (n x n)\n")
                        new_lines.append("    - alpha: tail probability (default 0.05 = 95% confidence level)\n")
                        new_lines.append("    - returns_data: DataFrame of historical returns (required for quantile regression)\n")
                        new_lines.append("    - use_quantile_regression: if True, use quantile regression; else use parametric approach\n")
                        new_lines.append("    \n")
                        new_lines.append("    Returns:\n")
                        new_lines.append("    - Portfolio CVaR value (scalar)\n")
                        new_lines.append("    \"\"\"\n")
                        new_lines.append("    # Ensure weights sum to 1\n")
                        new_lines.append("    weights = np.array(weights)\n")
                        new_lines.append("    if abs(weights.sum() - 1.0) > 1e-6:\n")
                        new_lines.append("        weights = weights / weights.sum()\n")
                        new_lines.append("    \n")
                        new_lines.append("    # Portfolio expected return\n")
                        new_lines.append("    portfolio_return = weights @ mu\n")
                        new_lines.append("    \n")
                        new_lines.append("    # Portfolio volatility\n")
                        new_lines.append("    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)\n")
                        new_lines.append("    \n")
                        new_lines.append("    if use_quantile_regression and returns_data is not None:\n")
                        new_lines.append("        # Paper method: Use quantile regression for VaR\n")
                        new_lines.append("        try:\n")
                        new_lines.append("            # Calculate historical portfolio returns with current weights\n")
                        new_lines.append("            portfolio_returns = (returns_data @ weights).values\n")
                        new_lines.append("            \n")
                        new_lines.append("            # Calculate VaR using quantile regression\n")
                        new_lines.append("            var = calculate_var_quantile_regression(portfolio_returns, alpha=alpha)\n")
                        new_lines.append("            \n")
                        new_lines.append("            # Calculate CVaR from VaR\n")
                        new_lines.append("            # CVaR = E[R | R <= VaR] = mean of returns below VaR\n")
                        new_lines.append("            tail_returns = portfolio_returns[portfolio_returns <= var]\n")
                        new_lines.append("            \n")
                        new_lines.append("            if len(tail_returns) > 0:\n")
                        new_lines.append("                cvar = tail_returns.mean()\n")
                        new_lines.append("            else:\n")
                        new_lines.append("                # Fallback: use parametric approach\n")
                        new_lines.append("                z_alpha = norm.ppf(alpha)\n")
                        new_lines.append("                phi_z = norm.pdf(z_alpha)\n")
                        new_lines.append("                cvar_multiplier = phi_z / (1 - alpha)\n")
                        new_lines.append("                cvar = portfolio_return + portfolio_vol * cvar_multiplier\n")
                        new_lines.append("            \n")
                        new_lines.append("            return cvar\n")
                        new_lines.append("            \n")
                        new_lines.append("        except Exception as e:\n")
                        new_lines.append("            # Fallback to parametric approach if quantile regression fails\n")
                        new_lines.append("            if verbose:\n")
                        new_lines.append("                print(f\"    Warning: Quantile regression failed, using parametric approach: {str(e)}\")\n")
                        new_lines.append("            use_quantile_regression = False\n")
                        new_lines.append("    \n")
                        new_lines.append("    # Parametric approach (fallback or if use_quantile_regression=False)\n")
                        new_lines.append("    # Paper Formula (Equation 2, Page 3) - parametric version:\n")
                        new_lines.append("    # CVaR_α = x^T μ + √(x^T Ω x) * φ(φ^(-1)(α)) / (1-α)\n")
                        new_lines.append("    z_alpha = norm.ppf(alpha)\n")
                        new_lines.append("    phi_z = norm.pdf(z_alpha)\n")
                        new_lines.append("    cvar_multiplier = phi_z / (1 - alpha)\n")
                        new_lines.append("    portfolio_cvar = portfolio_return + portfolio_vol * cvar_multiplier\n")
                        new_lines.append("    \n")
                        new_lines.append("    return portfolio_cvar\n")
                        
                        # Skip the old function definition
                        i += 1
                        while i < len(source_lines) and not source_lines[i].strip().startswith('def ') and not source_lines[i].strip().startswith('# Test'):
                            i += 1
                        continue
                    
                    new_lines.append(line)
                    i += 1
                
                if modified:
                    cell['source'] = new_lines
                    fixes_applied += 1
                    print(f"  [OK] Added quantile regression for VaR calculation")
            
            # Update optimize_cvar_rp to pass returns_data to calculate_portfolio_cvar
            if 'def optimize_cvar_rp' in source_text and 'returns_data' in source_text:
                print(f"\n[Cell {cell_idx}] Updating optimize_cvar_rp to use quantile regression...")
                
                if isinstance(source, list):
                    source_lines = source
                else:
                    source_lines = source.split('\n')
                    source_lines = [line + '\n' if i < len(source_lines) - 1 else line 
                                   for i, line in enumerate(source_lines)]
                
                new_lines = []
                modified = False
                
                for i, line in enumerate(source_lines):
                    # Find where calculate_portfolio_cvar is called (if it is)
                    if 'calculate_portfolio_cvar' in line and 'returns_data' not in line:
                        # Update to pass returns_data
                        if 'portfolio_cvar = calculate_portfolio_cvar' in line or 'cvar = calculate_portfolio_cvar' in line:
                            # Extract the function call
                            if 'portfolio_cvar =' in line:
                                new_line = line.replace(
                                    'calculate_portfolio_cvar(weights, mu, cov_matrix, alpha=alpha)',
                                    'calculate_portfolio_cvar(weights, mu, cov_matrix, alpha=alpha, returns_data=returns_data, use_quantile_regression=True)'
                                )
                                new_lines.append(new_line)
                                modified = True
                                continue
                    
                    new_lines.append(line)
                
                if modified:
                    cell['source'] = new_lines
                    fixes_applied += 1
                    print(f"  [OK] Updated optimize_cvar_rp to use quantile regression")
    
    if fixes_applied > 0:
        print(f"\nWriting updated notebook...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"[OK] Applied {fixes_applied} fixes to add quantile regression for VaR")
        return True
    else:
        print("No fixes needed or cells not found")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    success = fix_quantile_regression_var(notebook_path)
    sys.exit(0 if success else 1)
