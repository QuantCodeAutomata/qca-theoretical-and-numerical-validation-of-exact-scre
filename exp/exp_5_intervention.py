"""
Experiment 5: Causal Intervention Experiment

n=40 assets, 250 replications.  Calls src.intervention.run_intervention_experiment
and reports max-absolute-residual-correlation diagnostics for:
  - Pre-intervention:  common-state vs proxy conditioning
  - Post-intervention: common-state vs proxy conditioning (proxy now broken)

Target values (from paper):
  Pre:  common ~0.212, proxy ~0.212
  Post: common ~0.211, proxy ~0.911
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intervention import run_intervention_experiment

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Experiment 5: Causal Intervention Experiment ===\n")
    print("  n_assets=40, n_replications=250, seed=42")
    print("  Running ... (this may take ~30-60 seconds)\n")

    results = run_intervention_experiment(n_assets=40, n_replications=250, seed=42)

    pre_c = results["pre_common_mean"]
    pre_p = results["pre_proxy_mean"]
    post_c = results["post_common_mean"]
    post_p = results["post_proxy_mean"]
    pre_c_se = results["pre_common_mcse"]
    pre_p_se = results["pre_proxy_mcse"]
    post_c_se = results["post_common_mcse"]
    post_p_se = results["post_proxy_mcse"]

    # ------------------------------------------------------------------ #
    # Print results table
    # ------------------------------------------------------------------ #
    print("  Results (max abs off-diagonal residual correlation):")
    print(f"  {'Condition':<30} {'Common-state':>16} {'Proxy':>16}")
    print(f"  {'-'*30} {'-'*16} {'-'*16}")
    print(f"  {'Pre-intervention':30} {pre_c:>16.4f}     {pre_p:>16.4f}")
    print(f"  {'  +-2*MCSE':30} {'±' + f'{2*pre_c_se:.4f}':>16}     {'±' + f'{2*pre_p_se:.4f}':>16}")
    print(f"  {'Post-intervention':30} {post_c:>16.4f}     {post_p:>16.4f}")
    print(f"  {'  +-2*MCSE':30} {'±' + f'{2*post_c_se:.4f}':>16}     {'±' + f'{2*post_p_se:.4f}':>16}")
    print()
    print("  Paper targets:")
    print("    Pre:  common ~0.212, proxy ~0.212")
    print("    Post: common ~0.211, proxy ~0.911")

    # ------------------------------------------------------------------ #
    # Save JSON
    # ------------------------------------------------------------------ #
    metrics = {
        "pre_common_mean": pre_c,
        "pre_proxy_mean": pre_p,
        "post_common_mean": post_c,
        "post_proxy_mean": post_p,
        "pre_common_mcse": pre_c_se,
        "pre_proxy_mcse": pre_p_se,
        "post_common_mcse": post_c_se,
        "post_proxy_mcse": post_p_se,
        "pre_common_ci95_lower": pre_c - 2 * pre_c_se,
        "pre_common_ci95_upper": pre_c + 2 * pre_c_se,
        "pre_proxy_ci95_lower": pre_p - 2 * pre_p_se,
        "pre_proxy_ci95_upper": pre_p + 2 * pre_p_se,
        "post_common_ci95_lower": post_c - 2 * post_c_se,
        "post_common_ci95_upper": post_c + 2 * post_c_se,
        "post_proxy_ci95_lower": post_p - 2 * post_p_se,
        "post_proxy_ci95_upper": post_p + 2 * post_p_se,
    }
    json_path = os.path.join(RESULTS_DIR, "exp5_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics JSON saved -> {json_path}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp5_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "model", "mean", "mcse", "ci95_lower", "ci95_upper"])
        writer.writerow(["pre", "common", pre_c, pre_c_se, pre_c - 2*pre_c_se, pre_c + 2*pre_c_se])
        writer.writerow(["pre", "proxy", pre_p, pre_p_se, pre_p - 2*pre_p_se, pre_p + 2*pre_p_se])
        writer.writerow(["post", "common", post_c, post_c_se, post_c - 2*post_c_se, post_c + 2*post_c_se])
        writer.writerow(["post", "proxy", post_p, post_p_se, post_p - 2*post_p_se, post_p + 2*post_p_se])
    print(f"  Metrics CSV saved -> {csv_path}")

    # ------------------------------------------------------------------ #
    # Bar chart with error bars
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Pre / Common", "Pre / Proxy", "Post / Common", "Post / Proxy"]
    means = [pre_c, pre_p, post_c, post_p]
    errors = [2 * pre_c_se, 2 * pre_p_se, 2 * post_c_se, 2 * post_p_se]
    colors = ["#4878CF", "#4878CF", "#D65F5F", "#D65F5F"]
    hatches = ["", "//", "", "//"]

    bars = ax.bar(labels, means, yerr=errors, color=colors, hatch=hatches,
                  capsize=6, alpha=0.85, error_kw={"linewidth": 1.5})

    ax.set_ylabel("Max abs off-diagonal residual correlation")
    ax.set_title(
        "Intervention Experiment: Max Residual Correlation\n"
        "Blue=pre, Red=post; solid=common-state, hatched=proxy"
    )
    ax.set_ylim(0, min(1.05, max(means) * 1.25))
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="Theoretical max")
    ax.legend(fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "exp5_intervention_barplot.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Bar chart saved -> {plot_path}")


if __name__ == "__main__":
    main()
