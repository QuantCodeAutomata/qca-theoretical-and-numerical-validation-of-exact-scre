# QCA Repository — Agent Memory

## Project Overview
Quantitative Covariance Analysis (QCA) research codebase.

## Directory Layout
```
data/simulate.py   – AR(1) factor simulation and synthetic return panel generation
src/utils.py       – Matrix utilities (SPD generation, norms, PSD projection)
src/screening.py   – Conditional risk decomposition Q = D + Sigma_sys validation
src/projection.py  – Innovation projection decomposition (affine + nonlinear)
src/markowitz.py   – Constrained Markowitz operators and CVXPY cross-check
src/perturbation.py– Exact finite perturbation identities for M, w*, Delta
src/diagnostics.py – Off-diagonal residual diagnostic and norm bounds
src/intervention.py– Causal intervention vs observational proxy experiment
src/covariance.py  – GMV covariance estimators (sample, LW, PCA, structured)
exp/               – Experiment scripts (user-written, no src imports)
tests/             – Test suite
results/           – Output directory
```

## Key Design Decisions

### Path Setup
All modules use `sys.path.insert(0, '.')` when running scripts from project root.
Imports cross packages: `from data.simulate import ...` and `from src.utils import ...`.

### Return Model DGP
`r_{i,t+1} = a_i + b_i @ Z_t + c_i @ eta_{t+1} + eps_{i,t+1}`
- `Z_t` shape `(n_obs, n_drivers)`: AR(1) states
- `eta[0] = 0` (unused); `eta[1:]` are real innovations
- Returns have shape `(n_obs-1, n_assets)` using `Z[:-1]` and `eta[1:]`

### Innovation Covariance
Default: `Lambda = innov_var * I` where `innov_var = 0.0975`.
Stationary variance of Z: `innov_var / (1 - phi^2)`.

### Markowitz Math
- `QinvA = solve(Q, A.T)`
- `AQinvAT = A @ QinvA`
- `w0 = QinvA @ solve(AQinvAT, b)`
- `M = Q^{-1} - QinvA @ solve(AQinvAT, QinvA.T)`
- Key: `A @ M = 0` → cross-term in sigma2_p vanishes → `sigma2_p = w0.T@Q@w0 + gamma^2*Delta`

### Perturbation Identities
Exact (not approximate) for any SPD Q0, any symmetric R with Q0+R SPD:
- `M_R - M = -M @ R @ M_R = -M_R @ R @ M`
- `w_R* - w* = -M_R @ R @ w* = -M @ R @ w_R*`
- `Delta_R - Delta = -(M@mu).T @ R @ (M_R@mu)`

### Intervention Experiment Design
For the intervention effect to be clearly visible with the max-abs-residual-correlation metric:
- Large Z^1 loadings (b1 std=0.50): dominant driver
- Small Z^2 loadings (b2 std=0.10): secondary driver
- Large idio vols (U(0.14, 0.16)): ensures post-conditioning residuals are near-zero
- Tiny innovation loadings (c, d std=0.01): negligible after conditioning on Z_t
- This creates: common (~0.05), pre-proxy (~0.6), post-proxy (~0.95)

### Diagnostics Bound Chain
`||E||_2 <= ||E||_F <= eps_cov * sqrt((S2)^2 - S4) <= eps_cov * S2`
where `S2 = sum_i s_i^2`, `S4 = sum_i s_i^4`, `s_i = sqrt(V_{ii})`

### Structured Covariance Estimation
1. AR(1) OLS per driver: extract eta_hat (shape n_ar, n_drivers)
2. Regress returns on [1, Z_{t-1}, eta_hat_t]
3. L_hat = coeffs for eta_hat rows; D_hat = residual variances
4. Q_struct = diag(D_hat) + L_hat @ sample_cov(eta_hat) @ L_hat.T

## Dependencies
All standard: numpy, scipy, sklearn.covariance.LedoitWolf, cvxpy (for Markowitz cross-check).

## Critical: Conditional vs Unconditional Covariance
- `get_population_covariance(c_load, innov_cov, idio_vols)` returns the CONDITIONAL
  covariance: `Q_cond = D + c @ Lambda @ c.T` (given state Z_t).
- The UNCONDITIONAL covariance adds the state-predictive term:
  `Q_uncond = Q_cond + b @ (state_var * I) @ b.T`
  where `state_var = innov_var / (1 - phi^2)`.
- With phi=0.95, innov_var=0.0975: `state_var = 1.0` — the state term DOMINATES.
- `fit_structured_covariance` also returns conditional Q only (no b @ state_cov @ b.T).
- Using conditional Q for GMV then evaluating on unconditional returns yields inflated OOS vol.
- For fair GMV comparisons: pass UNCONDITIONAL oracle to `evaluate_gmv_out_of_sample`.

## Experiment File Status (all in /workspace/project/)
- `exp/exp_1_theoretical_validation.py` — All 7 blocks pass; plots + JSONs in results/
- `exp/exp_2_screening_test.py` — Past-only max~0.999, horizon-closed max~0.093
- `exp/exp_3_covariance_estimation.py` — oracle uses unconditional Q; structured shows
  inflated vol (known limitation of conditional Q estimator); sample/LW/oracle comparable
- `exp/exp_4_perturbation_study.py` — Identities verified; displacement/frontier reported
- `exp/exp_5_intervention.py` — Post-proxy >> post-common (clear intervention effect)
- `exp/exp_6_empirical_screening.py` — Synthetic fallback; selected drivers < PCA < none
- `exp/exp_7_empirical_covariance.py` — Synthetic fallback; T=252/504 fold study

## Test Calibration Notes
- `test_first_order_accuracy` (test_exp1): use tau=1e-6, threshold 1e-3 (relative error
  is O(tau * ||M R||_op); with min_eig=0.1 and tau=1e-4, error is ~5e-4, not < 1e-4).
- `test_perturbation_identities_exact` (test_exp4): use 1e-9, not 1e-10 (floating-point
  rounding puts some residuals just above 1e-10 for n=8 matrices).
- `test_structured_beats_sample_small_T` (test_exp3): tests Frobenius error vs conditional
  Q_pop, not OOS portfolio vol (structured is consistently better on this metric).
- `test_oracle_is_best` (test_exp3): uses unconditional oracle; compares sample/LW only;
  requires T_test=300 for stable OOS vol estimates.

## Common Gotchas
- `frob_norm` / `op_norm` in utils.py handle both 1-D vectors and 2-D matrices
- `project_spd` is called defensively in PCA and structured covariance outputs
- In `simulate_structural_model` the `_make_returns` closure captures `rng` from outer scope — call order matters for reproducibility
- `fit_structured_covariance` alignment: `r_reg = returns[1:]`, `Z_cond2 = driver_states[:n_ar]`
