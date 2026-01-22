"""
Debug script to add to the notebook before running Kupiec Test.
Run this cell right before the Kupiec Test to diagnose the issue.
"""

print("=" * 80)
print("DEBUGGING KUPIEC TEST DATE ALIGNMENT")
print("=" * 80)

# Check if cvar_rp_results exists
try:
    print(f"\n1. cvar_rp_results exists: {len(cvar_rp_results)} rows")
    print(f"   Columns: {list(cvar_rp_results.columns)}")
    print(f"   Date range: {cvar_rp_results['date'].min()} to {cvar_rp_results['date'].max()}")
    print(f"   Date type: {type(cvar_rp_results['date'].iloc[0])}")
    print(f"\n   First 3 dates in cvar_rp_results:")
    for i in range(min(3, len(cvar_rp_results))):
        print(f"     {cvar_rp_results['date'].iloc[i]} (type: {type(cvar_rp_results['date'].iloc[i])})")
except NameError:
    print("\n✗ ERROR: cvar_rp_results is not defined!")
    print("   You need to run Cell 19 (Phase 2.4 - Baseline CVaR-RP Backtest) first")
except Exception as e:
    print(f"\n✗ ERROR checking cvar_rp_results: {e}")

# Check if returns_val and returns_test exist
try:
    print(f"\n2. returns_val exists: {len(returns_val)} rows")
    print(f"   Date range: {returns_val.index.min()} to {returns_val.index.max()}")
except NameError:
    print("\n✗ ERROR: returns_val is not defined!")

try:
    print(f"\n3. returns_test exists: {len(returns_test)} rows")
    print(f"   Date range: {returns_test.index.min()} to {returns_test.index.max()}")
except NameError:
    print("\n✗ ERROR: returns_test is not defined!")

# Check backtest_data
try:
    backtest_data = pd.concat([returns_val, returns_test])
    print(f"\n4. backtest_data (concat): {len(backtest_data)} rows")
    print(f"   Date range: {backtest_data.index.min()} to {backtest_data.index.max()}")
    print(f"   Index type: {type(backtest_data.index)}")
except Exception as e:
    print(f"\n✗ ERROR creating backtest_data: {e}")

print("\n" + "=" * 80)
print("If all checks pass, the Kupiec Test should work.")
print("If cvar_rp_results is not defined, run Cell 19 first.")
print("=" * 80)
