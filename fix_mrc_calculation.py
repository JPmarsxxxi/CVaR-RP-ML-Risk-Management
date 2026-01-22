#!/usr/bin/env python3
"""
Script to fix the Marginal Risk Contribution (MRC) calculation in the notebook.
Fixes both the Numba JIT version and the standard Python version.
"""

import json
import sys

def fix_notebook_mrc_calculation(notebook_path):
    """Fix MRC calculation in both Numba and standard Python versions."""
    
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
            
            # Check if this cell contains the MRC calculation
            source_text = ''.join(source_lines) if isinstance(source_lines, list) else source_lines
            
            # Fix Numba JIT version
            if 'def _cvar_rp_core' in source_text and '# Beta (marginal risk contribution)' in source_text:
                print(f"\nFound Numba JIT version in cell {cell_idx}")
                new_lines = []
                i = 0
                while i < len(source_lines):
                    line = source_lines[i]
                    # Find and replace the MRC calculation block
                    if '# Beta (marginal risk contribution)' in line:
                        new_lines.append('            # Marginal Risk Contribution (MRC) - Paper Formula (Equation 3):\n')
                        new_lines.append('            # MRC_i(x) = r_i + (Ωx)_i / √(x^T Ω x) * φ(φ^(-1)(α))/(1-α)\n')
                        new_lines.append('            # where r_i = μ_i (mean return of asset i)\n')
                        new_lines.append('            # (Ωx)_i is the i-th element of the vector Ωx\n')
                        new_lines.append('            # √(x^T Ω x) is portfolio volatility\n')
                        new_lines.append('            # φ(φ^(-1)(α))/(1-α) is the CVaR multiplier\n')
                        i += 1
                        # Skip old calculation lines
                        while i < len(source_lines) and any(x in source_lines[i] for x in [
                            'beta = np.zeros', 'for i in range(n)', 'cov_x = 0.0',
                            'for j in range(n)', 'cov_x += cov_matrix', 'beta[i] = mu[i] + (cov_x / portfolio_vol)'
                        ]):
                            i += 1
                        # Add corrected calculation
                        new_lines.append('            beta = np.zeros(n)\n')
                        new_lines.append('            for i in range(n):\n')
                        new_lines.append('                # Compute (Ωx)_i: i-th element of covariance matrix @ weights\n')
                        new_lines.append('                cov_x_i = 0.0\n')
                        new_lines.append('                for j in range(n):\n')
                        new_lines.append('                    cov_x_i += cov_matrix[i, j] * x[j]\n')
                        new_lines.append('                # MRC_i = r_i + (Ωx)_i / √(x^T Ω x) * CVaR_multiplier\n')
                        new_lines.append('                beta[i] = mu[i] + (cov_x_i / portfolio_vol) * cvar_multiplier\n')
                        continue
                    new_lines.append(line)
                    i += 1
                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Fixed Numba JIT version")
            
            # Fix standard Python version
            elif '# Calculate beta (marginal risk contribution)' in source_text:
                print(f"\nFound standard Python version in cell {cell_idx}")
                new_lines = []
                i = 0
                while i < len(source_lines):
                    line = source_lines[i]
                    # Find and replace the MRC calculation
                    if '# Calculate beta (marginal risk contribution)' in line:
                        new_lines.append('            # Calculate Marginal Risk Contribution (MRC) - Paper Formula (Equation 3, Page 3):\n')
                        new_lines.append('            # MRC_i(x) = ∂(CVaR)/∂x_i = r_i + (Ωx)_i / √(x^T Ω x) * φ(φ^(-1)(α))/(1-α)\n')
                        new_lines.append('            # where:\n')
                        new_lines.append('            #   r_i = μ_i (mean return of asset i) - in code: mu[i]\n')
                        new_lines.append('            #   (Ωx)_i = i-th element of covariance matrix @ weights\n')
                        new_lines.append('            #   √(x^T Ω x) = portfolio volatility\n')
                        new_lines.append('            #   φ(φ^(-1)(α))/(1-α) = CVaR multiplier\n')
                        new_lines.append('            # \n')
                        new_lines.append('            # Compute (Ωx) explicitly: covariance matrix @ weights\n')
                        new_lines.append('            cov_x = cov_matrix @ x  # This is the vector (Ωx)\n')
                        new_lines.append('            # MRC_i = r_i + (Ωx)_i / √(x^T Ω x) * CVaR_multiplier\n')
                        new_lines.append('            beta = mu + (cov_x / portfolio_vol) * cvar_multiplier\n')
                        i += 1
                        # Skip old lines
                        while i < len(source_lines) and any(x in source_lines[i] for x in [
                            '# Beta_i = mu_i', 'beta = mu + (cov_matrix @ x)'
                        ]):
                            i += 1
                        continue
                    new_lines.append(line)
                    i += 1
                cell['source'] = new_lines
                fixes_applied += 1
                print("  [OK] Fixed standard Python version")
    
    if fixes_applied > 0:
        print(f"\n[OK] Applied {fixes_applied} fix(es)")
        print(f"Writing fixed notebook to: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook fixed successfully!")
        return True
    else:
        print("\n[WARNING] No fixes applied. The MRC calculation might already be correct or the pattern didn't match.")
        return False

if __name__ == '__main__':
    notebook_path = 'CVaR_RP_ML_Implementation.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    
    try:
        fix_notebook_mrc_calculation(notebook_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
