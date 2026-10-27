"""
OASIS-2 mini prognostic model: can baseline measures predict who will decline fast?

Purpose: get a REALISTIC (and honestly uncertain) AUC to plug into the
stratified-trial simulation (stratified_trial_sim.py).

Steps:
  1. One row per subject: baseline measures (Visit 1).
  2. Outcome: each subject's MMSE slope per year across all their visits.
  3. Cohort: early disease only (baseline CDR <= 0.5), >= 2 visits.
  4. Label "fast progressor" = steepest-declining third.
  5. Logistic regression, repeated stratified cross-validation, AUC reported
     with its spread. Because each subject is ONE row, there is no
     within-person leakage.

Run:  python oasis_prognostic.py --csv path/to/oasis_longitudinal.csv

Column names are taken from published descriptions of the OASIS-2 CSV:
Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand, Age, EDUC, SES,
MMSE, CDR, eTIV, nWBV, ASF. Check them against your file. 'MR Delay' is assumed
to be days since the first visit; verify that too.
Download OASIS data under its Data Usage Agreement.

Status: tested on a synthetic CSV with these columns, not yet on the real file.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_ALL = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "Male"]


def build_subject_table(path):
    df = pd.read_csv(path)
    df = df.sort_values(["Subject ID", "Visit"])
    rows = []
    for sid, g in df.groupby("Subject ID"):
        g = g.dropna(subset=["MMSE"])
        if len(g) < 2:
            continue
        years = g["MR Delay"].to_numpy() / 365.25
        if years.max() <= 0:
            continue
        slope = np.polyfit(years, g["MMSE"].to_numpy(), 1)[0]
        base = g.iloc[0]
        rows.append({"Subject ID": sid, "slope": slope, "Age": base["Age"],
                     "EDUC": base["EDUC"], "SES": base["SES"], "MMSE": base["MMSE"],
                     "CDR": base["CDR"], "eTIV": base["eTIV"], "nWBV": base["nWBV"],
                     "ASF": base["ASF"], "Male": 1 if str(base["M/F"]).strip().upper().startswith("M") else 0})
    return pd.DataFrame(rows)


def cv_auc(X, y, n_repeats=20, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    aucs = []
    for train, test in cv.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        model.fit(X[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
    return np.mean(aucs), np.percentile(aucs, [5, 95])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    tbl = build_subject_table(args.csv)
    tbl = tbl[tbl["CDR"] <= 0.5].copy()
    tbl["SES"] = tbl["SES"].fillna(tbl["SES"].median())
    cutoff = tbl["slope"].quantile(1 / 3)
    tbl["fast"] = (tbl["slope"] <= cutoff).astype(int)
    print(f"Subjects in cohort: {len(tbl)} | fast progressors: {tbl['fast'].sum()} "
          f"(MMSE slope <= {cutoff:.2f} points/year)")

    y = tbl["fast"].to_numpy()
    for name, feats in [("all baseline features", FEATURES_ALL),
                        ("without baseline MMSE and CDR", [f for f in FEATURES_ALL if f not in ("MMSE", "CDR")])]:
        mean_auc, (lo, hi) = cv_auc(tbl[feats].to_numpy(dtype=float), y)
        print(f"{name:<32} AUC = {mean_auc:.2f}  (5th-95th pct of CV folds: {lo:.2f}-{hi:.2f})")

    print("\nCaveats: small sample, wide uncertainty. Fold AUCs overlap so the range "
          "is not a true confidence interval. Use a RANGE of AUCs in the simulation, "
          "not a single number.")