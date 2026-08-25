"""
Tests for Experiment 5: Causal Intervention.

Uses n_assets=8 and n_replications=10 for speed (< 30 s).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.intervention import run_intervention_experiment


def _run_small(seed: int = 42):
    """Run the intervention experiment with small parameters for testing."""
    return run_intervention_experiment(n_assets=8, n_replications=10, seed=seed)


# ------------------------------------------------------------------ #
# Test 1: Pre-intervention equivalence
# ------------------------------------------------------------------ #

def test_pre_intervention_equivalence():
    """
    Pre-intervention: common-state and proxy diagnostics should be close,
    since the proxy is a deterministic linear function of the causal state.
    Tolerance: difference < 0.5 (liberal for n_replications=10).
    """
    res = _run_small()
    diff = abs(res["pre_common_mean"] - res["pre_proxy_mean"])
    assert diff < 0.5, (
        f"Pre-intervention common={res['pre_common_mean']:.4f} and "
        f"proxy={res['pre_proxy_mean']:.4f} differ by {diff:.4f} > 0.5"
    )


# ------------------------------------------------------------------ #
# Test 2: Post-intervention divergence
# ------------------------------------------------------------------ #

def test_post_intervention_divergence():
    """
    Post-intervention: proxy diagnostic should exceed common-state diagnostic
    by at least 0.3 (proxy is now independent of the causal state).
    """
    res = _run_small()
    gap = res["post_proxy_mean"] - res["post_common_mean"]
    assert gap > 0.3, (
        f"Post-intervention gap (proxy - common) = {gap:.4f}, expected > 0.3. "
        f"post_common={res['post_common_mean']:.4f}, post_proxy={res['post_proxy_mean']:.4f}"
    )


# ------------------------------------------------------------------ #
# Test 3: Common-state diagnostic invariant across regimes
# ------------------------------------------------------------------ #

def test_common_state_invariant():
    """
    Common-state model should be roughly stable across pre and post regimes
    (it conditions on the actual causal state, which is structurally unchanged).
    """
    res = _run_small()
    diff = abs(res["post_common_mean"] - res["pre_common_mean"])
    assert diff < 0.4, (
        f"Common-state diagnostic changed by {diff:.4f} across regimes "
        f"(pre={res['pre_common_mean']:.4f}, post={res['post_common_mean']:.4f})"
    )


# ------------------------------------------------------------------ #
# Test 4: All MCSE values positive
# ------------------------------------------------------------------ #

def test_mcse_positive():
    """All Monte Carlo standard errors should be strictly positive."""
    res = _run_small()
    for key in ("pre_common_mcse", "pre_proxy_mcse", "post_common_mcse", "post_proxy_mcse"):
        val = res[key]
        assert val > 0, f"{key}={val:.6f} is not positive"
        assert np.isfinite(val), f"{key}={val} is not finite"
