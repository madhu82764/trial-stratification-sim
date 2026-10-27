"""
How accurate must a prognostic score be before stratifying a trial pays off?

Setup (all numbers are ASSUMPTIONS you should vary and justify):
  - Patients are either slow or fast progressors (hidden). Outcome = change in a
    cognitive score over 18 months (negative = decline).
  - A prognostic score tries to identify slow progressors. Its quality is set by
    its AUC (0.5 = useless, 1.0 = perfect).
  - Scenario 1: drug helps ONLY slow progressors (the "treat early" story).
  - Scenario 2: drug helps everyone proportionally (bigger absolute gain in
    fast decliners).

Three analysis strategies, compared at the SAME number of enrolled patients:
  unstratified : all enrolled, simple t-test
  enriched     : screen more patients, enrol only those the score predicts to be
                 slow (Scenario 1) or fast (Scenario 2) progressors
  adjusted     : same cohort as unstratified, regression adjusting for the score

Run:  python stratified_trial_sim.py            (full run, a minute or two)
      python stratified_trial_sim.py --null     (sanity check: drug effect = 0,
                                                 false-positive rate should be ~5%)
"""
import argparse

import numpy as np
from scipy.stats import norm, t as t_dist, ttest_ind

AUCS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def auc_to_d(auc):
    """Separation between score distributions that gives the requested AUC
    (assumes equal-variance normal scores)."""
    return np.sqrt(2) * norm.ppf(auc)


def make_patients(n, d, rng, p_slow=0.5):
    slow = rng.random(n) < p_slow
    score = d * slow + rng.normal(size=n)  # higher score = predicted slow
    return slow, score


def outcomes(slow, trt, rng, mean_slow, mean_fast, sd, eff_slow, eff_fast):
    base = np.where(slow, mean_slow, mean_fast)
    effect = trt * np.where(slow, eff_slow, eff_fast)
    return base + effect + rng.normal(0, sd, size=len(slow))


def ols_trt_p(y, trt, score):
    """p-value for the treatment coefficient in y ~ 1 + trt + score."""
    X = np.column_stack([np.ones(len(y)), trt, score])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    t_stat = beta[1] / np.sqrt(cov[1, 1])
    return 2 * t_dist.sf(abs(t_stat), dof), beta[1]


def one_trial(n_enrol, auc, rng, cfg, screen_frac):
    d = auc_to_d(auc)
    results = {}

    # unstratified + adjusted use the same randomly enrolled cohort
    slow, score = make_patients(n_enrol, d, rng, cfg["p_slow"])
    trt = rng.permutation(np.arange(n_enrol) % 2)
    y = outcomes(slow, trt, rng, **cfg["outcome"])
    res = ttest_ind(y[trt == 1], y[trt == 0])
    results["unstratified"] = (res.pvalue < 0.05) and (y[trt == 1].mean() > y[trt == 0].mean())
    p, b = ols_trt_p(y, trt, score)
    results["adjusted"] = (p < 0.05) and (b > 0)

    # enrichment: screen n_enrol / screen_frac patients, keep the top n_enrol
    n_screen = int(round(n_enrol / screen_frac))
    slow_s, score_s = make_patients(n_screen, d, rng, cfg["p_slow"])
    order = np.argsort(score_s)  # higher score = predicted slow progressor
    keep = order[-n_enrol:] if cfg["enrich_for"] == "slow" else order[:n_enrol]
    slow_e = slow_s[keep]
    trt_e = rng.permutation(np.arange(n_enrol) % 2)
    y_e = outcomes(slow_e, trt_e, rng, **cfg["outcome"])
    res = ttest_ind(y_e[trt_e == 1], y_e[trt_e == 0])
    results["enriched"] = (res.pvalue < 0.05) and (y_e[trt_e == 1].mean() > y_e[trt_e == 0].mean())
    return results


def power_curve(n_enrol, cfg, reps, seed, screen_frac):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("unstratified", "enriched", "adjusted")}
    for auc in AUCS:
        counts = {k: 0 for k in out}
        for _ in range(reps):
            r = one_trial(n_enrol, auc, rng, cfg, screen_frac)
            for k in counts:
                counts[k] += r[k]
        for k in out:
            out[k].append(counts[k] / reps)
    return out


def scenario(eff_slow, eff_fast, enrich_for="slow"):
    return {
        "enrich_for": enrich_for,
        "p_slow": 0.5,
        "outcome": dict(mean_slow=-1.5, mean_fast=-4.5, sd=1.5,
                        eff_slow=eff_slow, eff_fast=eff_fast),
    }


def plot(curves, path="stratified_trial_power.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, res) in zip(axes, curves.items()):
        for name, vals in res.items():
            ax.plot(AUCS, vals, marker="o", label=name)
        ax.axhline(0.8, linestyle="--", color="gray", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Prognostic score AUC")
    axes[0].set_ylabel("Power (share of simulated trials that detect the drug)")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print("Saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true", help="drug has no effect: check false-positive rate")
    ap.add_argument("--n", type=int, default=240, help="enrolled patients per trial")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--enrich-s2", choices=["slow", "fast"], default="fast",
                    help="which group the enriched design selects in Scenario 2")
    ap.add_argument("--screen-frac", type=float, default=0.5,
                    help="share of screened patients enrolled in the enriched design")
    args = ap.parse_args()

    if args.null:
        cfg = scenario(0.0, 0.0)
        res = power_curve(args.n, cfg, args.reps, seed=1, screen_frac=args.screen_frac)
        print("False-positive rate with NO drug effect (target ~0.025 one-sided):")
        for k, v in res.items():
            print(f"  {k:<13}", [round(float(x), 3) for x in v])
    else:
        curves = {
            "Scenario 1: drug helps only slow progressors":
                power_curve(args.n, scenario(0.6, 0.0), args.reps, 1, args.screen_frac),
            f"Scenario 2: drug helps everyone proportionally (enrich for {args.enrich_s2})":
                power_curve(args.n, scenario(0.375, 1.125, args.enrich_s2), args.reps, 2, args.screen_frac),
        }
        print(f"n enrolled = {args.n}; enriched design screens {1/args.screen_frac:.1f}x as many patients")
        for title, res in curves.items():
            print("\n" + title)
            print(f"{'AUC':<8}" + "".join(f"{k:<15}" for k in res))
            for i, auc in enumerate(AUCS):
                print(f"{auc:<8}" + "".join(f"{res[k][i]:<15.2f}" for k in res))
        plot(curves)