"""
rPPG / non-rPPG dual 3-class model ensemble permutation importance.

Two XGBoost models (different feature subsets) are fused by averaging
predict_proba, then sklearn permutation_importance is run on all 7 features.

Usage:
    python permutation_importance_dual.py \\
        --csv results/unified_features_20260811_202802.csv \\
        --model-non xgb_model_3class_non-rppg.joblib \\
        --model-rppg xgb_model_3class_rppg.joblib
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

from train_classifier import FEATURE_COLS, build_class_maps, encode_labels, load_and_label


class DualModelEnsemble(BaseEstimator, ClassifierMixin):
    """Fuse two pre-trained 3-class models via probability averaging."""

    def __init__(self, bundle_non: dict, bundle_rppg: dict):
        self.bundle_non = bundle_non
        self.bundle_rppg = bundle_rppg
        self.feature_cols_ = FEATURE_COLS
        self.class_names_ = bundle_non["class_names"]
        self.model_non_ = bundle_non["model"]
        self.model_rppg_ = bundle_rppg["model"]
        self.cols_non_ = bundle_non["feature_cols"]
        self.cols_rppg_ = bundle_rppg["feature_cols"]
        self.classes_ = np.arange(len(self.class_names_))

    def fit(self, X, y=None):
        return self

    def _split(self, X: np.ndarray):
        idx = {c: i for i, c in enumerate(self.feature_cols_)}
        return (
            X[:, [idx[c] for c in self.cols_non_]],
            X[:, [idx[c] for c in self.cols_rppg_]],
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x_non, x_rppg = self._split(X)
        return (self.model_non_.predict_proba(x_non) + self.model_rppg_.predict_proba(x_rppg)) / 2.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))


def main():
    parser = argparse.ArgumentParser(description="Dual rPPG/non-rPPG ensemble permutation importance")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-non", default="xgb_model_3class_non-rppg.joblib")
    parser.add_argument("--model-rppg", default="xgb_model_3class_rppg.joblib")
    parser.add_argument("--out", default="results/permutation_importance_dual_rppg_nonrppg.csv")
    parser.add_argument("--n-repeats", type=int, default=30)
    args = parser.parse_args()

    bundle_non = joblib.load(args.model_non)
    bundle_rppg = joblib.load(args.model_rppg)
    class_names = bundle_non["class_names"]

    df = load_and_label(args.csv)
    X = df[FEATURE_COLS].values.astype(float)
    y = df["group"].values
    class_to_idx, _ = build_class_maps(class_names)
    y_idx = encode_labels(y, class_to_idx)

    ensemble = DualModelEnsemble(bundle_non, bundle_rppg)
    y_pred_idx = ensemble.predict(X)
    baseline_acc = accuracy_score(y_idx, y_pred_idx)
    baseline_f1 = f1_score(y_idx, y_pred_idx, average="macro")

    print("=== Dual model ensemble ===")
    print(f"CSV: {args.csv} ({len(df)} rows)")
    print(f"non-rPPG features: {bundle_non['feature_cols']}")
    print(f"rPPG features:     {bundle_rppg['feature_cols']}")
    print(f"Fusion: predict_proba mean")
    print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Baseline macro F1:   {baseline_f1:.4f}")

    for label, bundle in [("non-rPPG", bundle_non), ("rPPG", bundle_rppg)]:
        xi = df[bundle["feature_cols"]].values.astype(float)
        acc = accuracy_score(y_idx, bundle["model"].predict(xi))
        print(f"  [{label} alone] accuracy: {acc * 100:.2f}%")

    print("\n=== Permutation Importance (macro F1 drop) ===")
    result = permutation_importance(
        ensemble,
        X,
        y_idx,
        n_repeats=args.n_repeats,
        random_state=42,
        scoring="f1_macro",
        n_jobs=1,
    )

    order = np.argsort(result.importances_mean)[::-1]
    rows = []
    for rank, i in enumerate(order, 1):
        mean = result.importances_mean[i]
        std = result.importances_std[i]
        print(f"  {rank}. {FEATURE_COLS[i]:<25s} {mean:+.4f} +/- {std:.4f}")
        rows.append(
            {
                "rank": rank,
                "feature": FEATURE_COLS[i],
                "importance_mean": mean,
                "importance_std": std,
                "model_subset": "non-rppg" if FEATURE_COLS[i] in bundle_non["feature_cols"] else "rppg",
            }
        )

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
