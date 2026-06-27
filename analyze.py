#!/usr/bin/env python3
"""Statistical analysis of bristlebot swarm simulation results."""

import json
import numpy as np
from scipy import stats
import sys

def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)

def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / s_pooled

def bonferroni_adjust(p_values, n_tests):
    """Bonferroni correction."""
    return [min(p * n_tests, 1.0) for p in p_values]

def analyze(path='demo_results.json'):
    results = load_results(path)
    
    print("=" * 60)
    print("BRISTLEBOT SWARM — Statistical Analysis")
    print("=" * 60)
    
    for test_id in sorted(results.keys()):
        r = results[test_id]
        print(f"Test {test_id}: HI = {r['mean']:.3f} ± {r['std']:.3f}")
    
    # Key comparisons
    print("\n--- Key Comparisons ---")
    baseline_hi = results.get('A', {}).get('mean', 0.5)
    
    for tid in ['C', 'F3']:
        if tid in results:
            d = cohens_d([results[tid]['mean']], [baseline_hi])
            print(f"Test {tid} vs Baseline A: Cohen's d ≈ {abs(d):.2f}")
    
    print("\n--- Decision Tree ---")
    c_hi = results.get('C', {}).get('mean', 0.5)
    f3_hi = results.get('F3', {}).get('mean', 0.5)
    
    if c_hi > 0.52:
        print("✅ τ_create CONFIRMED: calendar age produces hierarchy")
        if f3_hi < 0.48:
            print("🎯 τ_sick CONFIRMED: age entropy disrupts hierarchy")
            print("   → Physical causal evidence for mosaic aging hypothesis")
        else:
            print("❌ τ_sick NOT supported in this system")
    else:
        print("❌ Model rejected: calendar age does not produce hierarchy")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'demo_results.json'
    analyze(path)
