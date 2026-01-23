#!/usr/bin/env python3
"""
Reconstruct the corrupted optimize_cvar_rp function in Cell 17.
The original cell only has function signatures with no body.
"""

import json
import sys

# The correct function implementation
CORRECT_FUNCTION = '''# ============================================================================
# PHASE 2.3: Iterative CVaR-RP Optimization
# ============================================================================

print("=" * 80)
print("PHASE 2.3: CVaR-RP OPTIMIZATION")
print("=" * 80)

def optimize_cvar_rp(returns_data, alpha=0.05, garch_volatilities=None, max_iter=3000, tol=1e-6, verbose=False):
    """
    CVaR Risk Parity Optimization using iterative algorithm.

    Paper Methodology (Section 3.2):
    Iteratively adjust portfolio weights so that each asset contributes equally
    to the portfolio's Conditional Value at Risk (CVaR).

    Algorithm:
    1. Start with equal weights (1/N for N assets)
    2. Calculate each asset's marginal contribution to portfolio CVaR
    3. Adjust weights: w_i ← w_i * (target_cvar / MRC_i)
    4. Normalize weights to sum to 1
    5. Repeat until convergence

    Parameters:
    - returns_data: DataFrame of returns (columns = assets, rows = time)
    - alpha: Confidence level (default 0.05 = 95% confidence)
    - garch_volatilities: dict of GARCH-forecasted variances {asset: variance} (optional)
      If provided, builds covariance matrix from GARCH vols + correlations
      If None, uses historical sample covariance (default)
    - max_iter: Maximum iterations (default 3000)
    - tol: Convergence tolerance (default 1e-6)
    - verbose: Print iteration details (default False)

    Returns:
    - Numpy array of optimized weights
    """
    # Step 1: Calculate mean returns and covariance matrix
    mu = returns_data.mean().values  # Mean returns (n,)

    # Build covariance matrix: use GARCH if provided, else sample covariance
    if garch_volatilities is not None:
        # Paper methodology: Build covariance from GARCH volatilities + correlations
        # Cov = diag(σ_garch) * Corr * diag(σ_garch)
        # This uses GARCH-forecasted volatilities instead of historical
        corr_matrix = returns_data.corr().values  # Correlation matrix
        # Get GARCH volatilities (variances) in same order as columns
        vol_vector = np.array([np.sqrt(garch_volatilities.get(asset, returns_data[asset].var()))
                                     for asset in returns_data.columns])
        # Build covariance: Cov_ij = σ_i * σ_j * Corr_ij
        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix
        if verbose:
            print(f"  Using GARCH-forecasted volatilities for covariance matrix")
            print(f"  GARCH vol range: {vol_vector.min():.4f} to {vol_vector.max():.4f}")
    else:
        # Fallback: use historical sample covariance
        cov_matrix = returns_data.cov().values  # Covariance matrix (n, n)
        if verbose:
            print(f"  Using historical sample covariance matrix")

    n = len(mu)  # Number of assets

    # Step 2: Initialize with equal weights
    weights = np.ones(n) / n

    if verbose:
        print(f"\\nStarting iterative CVaR-RP optimization...")
        print(f"  Assets: {n}")
        print(f"  Alpha: {alpha}")
        print(f"  Max iterations: {max_iter}")

    # Step 3: Iterative optimization
    for iteration in range(max_iter):
        # Calculate portfolio-level CVaR
        portfolio_cvar = calculate_portfolio_cvar(weights, mu, cov_matrix, alpha=alpha)

        # Calculate marginal risk contributions (MRC)
        asset_cvars = []
        for i in range(n):
            # Get CVaR for this asset individually
            w_i = np.zeros(n)
            w_i[i] = 1.0
            asset_cvar = calculate_portfolio_cvar(w_i, mu, cov_matrix, alpha=alpha)
            asset_cvars.append(asset_cvar)

        asset_cvars = np.array(asset_cvars)

        # Marginal Risk Contribution = weight * asset_cvar
        mrc = weights * asset_cvars

        # Target CVaR contribution (equal for all assets)
        target_cvar = portfolio_cvar / n

        # Update weights: w_i ← w_i * (target / MRC_i)
        # Avoid division by zero
        mrc_safe = np.where(np.abs(mrc) < 1e-10, 1e-10, mrc)
        weights_new = weights * (target_cvar / mrc_safe)

        # Normalize weights to sum to 1
        weights_new = weights_new / weights_new.sum()

        # Check for convergence
        weight_change = np.abs(weights_new - weights).max()

        if verbose and iteration % 500 == 0:
            print(f"  Iteration {iteration}: max weight change = {weight_change:.6f}")

        # Update weights
        weights = weights_new

        # Check convergence
        if weight_change < tol:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations")
            break

    if verbose:
        print(f"\\nFinal weights:")
        for i, asset in enumerate(returns_data.columns):
            print(f"    {asset}: {weights[i]:.4f}")

    return weights

# Test on training data
print("\\n[Step 1] Testing CVaR-RP optimization on training data...")
cvar_rp_weights = optimize_cvar_rp(returns_train, alpha=0.05, verbose=True)

print("\\nOptimized CVaR-RP Weights:")
for i, ticker in enumerate(returns_train.columns):
    print(f"  {ticker}: {cvar_rp_weights[i]:.4f} ({cvar_rp_weights[i]*100:.2f}%)")

print("\\n" + "=" * 80)
print("PHASE 2.3 COMPLETE")
print("=" * 80)
'''

def fix_corrupted_cell17(notebook_path):
    """Replace the corrupted Cell 17 with proper function implementation."""

    print(f"Reading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Replace Cell 17
    if len(notebook['cells']) > 17 and notebook['cells'][17]['cell_type'] == 'code':
        source = ''.join(notebook['cells'][17]['source']) if isinstance(notebook['cells'][17]['source'], list) else notebook['cells'][17]['source']

        if 'PHASE 2.3' in source:
            print(f"\\nFound corrupted Cell 17 (Phase 2.3)")
            print(f"  Current cell has {len(source.split(chr(10)))} lines")
            print(f"  Replacing with proper function implementation...")

            # Replace with correct implementation
            notebook['cells'][17]['source'] = CORRECT_FUNCTION.split('\n')
            # Add newlines
            notebook['cells'][17]['source'] = [line + '\n' if i < len(notebook['cells'][17]['source']) - 1 else line
                                                for i, line in enumerate(notebook['cells'][17]['source'])]

            print(f"  [OK] Replaced Cell 17 with proper implementation")
            print(f"  New cell has {len(CORRECT_FUNCTION.split(chr(10)))} lines")

            # Save
            print(f"\\nWriting fixed notebook to: {notebook_path}")
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            print("[OK] Notebook fixed successfully!")
            return True
        else:
            print("\\n[WARNING] Cell 17 doesn't contain PHASE 2.3")
            return False
    else:
        print("\\n[ERROR] Cell 17 not found or not a code cell")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    try:
        success = fix_corrupted_cell17(notebook_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
