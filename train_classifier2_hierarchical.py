"""
계층적 2단계 분류 파이프라인 (XGBoost + SHAP)
==============================================

1단계: real vs fake
2단계: 1단계에서 fake 로 판단된 영상만 deepfake vs diffusion

train_classifier2.py 와 같은 데이터 / 10개 피처 / 라벨 규칙 / train-test 분리를 쓴다.
train_classifier2.py 자체는 수정하지 않는다.

사용법:
    python train_classifier2_hierarchical.py --csv results/unified_features_20260911_024351.csv
    python train_classifier2_hierarchical.py --csv ... --compare-dir results/train_classifier2_3진분류
    python train_classifier2_hierarchical.py --csv ... --tune --compare-dir results/train_classifier2_3진분류
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_classifier2 import (
    CLASS_NAMES,
    DEFAULT_REAL_DIR,
    DISPLAY_LABEL,
    DISPLAY_ORDER,
    FEATURE_COLS,
    Tee,
    _bullets,
    _md_table,
    _sample_weights,
    build_class_maps,
    classify_subgroup,
    compute_metrics,
    decode_labels,
    encode_labels,
    load_and_label,
    metrics_table,
    parse_class_weight,
    per_class_table,
    remove_outliers_iqr,
)

warnings.filterwarnings("ignore")

STAGE1_NAMES = ["real", "fake"]
STAGE2_NAMES = ["diffusion", "deepfake"]
STAGE1_DISPLAY = {"real": "Real", "fake": "Fake"}
STAGE2_DISPLAY = {"deepfake": "Deepfake", "diffusion": "Diffusion"}
STAGE1_ORDER = ["real", "fake"]
STAGE2_ORDER = ["deepfake", "diffusion"]

PHYS_FEATURES = ["bvp_std", "d3_temporal_score", "highfreq_score"]
FACE_FEATURES = [
    "identity_sim_std",
    "boundary_score_mean",
    "boundary_score_std",
    "boundary_score_max",
]
PATCH_FEATURES = ["absdiff_std", "patch_corr_mean", "patch_signal_std_mean"]
TUNE_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
TUNE_FAKE_WEIGHTS = (1.0, 1.5, 2.0)


def feature_presets() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "all": (list(FEATURE_COLS), list(FEATURE_COLS)),
        "phys_vs_face": (list(PHYS_FEATURES), list(FACE_FEATURES)),
        "phys_id_vs_face": (PHYS_FEATURES + ["identity_sim_std"], list(FACE_FEATURES)),
        "phys_patch_vs_face": (PHYS_FEATURES + PATCH_FEATURES, list(FACE_FEATURES)),
        "phys_vs_all": (list(PHYS_FEATURES), list(FEATURE_COLS)),
        "all_vs_face": (list(FEATURE_COLS), list(FACE_FEATURES)),
        "all_vs_face_bvp": (list(FEATURE_COLS), FACE_FEATURES + ["bvp_std"]),
        "top_split": (
            ["identity_sim_std", "bvp_std", "highfreq_score", "boundary_score_mean", "boundary_score_std"],
            ["bvp_std", "identity_sim_std", "boundary_score_std", "boundary_score_mean", "highfreq_score"],
        ),
    }


def parse_features(spec: str | None) -> list[str]:
    if not spec or spec.strip() in ("all", "*"):
        return list(FEATURE_COLS)
    cols = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in cols if c not in FEATURE_COLS]
    if unknown:
        raise SystemExit(f"알 수 없는 피처: {unknown}. 사용 가능: {FEATURE_COLS}")
    return cols


def slice_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 없는 피처: {missing}")
    return df[cols].values.astype(float)


def to_binary_labels(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) == "real", "real", "fake")


def _make_binary_xgb(random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        missing=np.nan,
        random_state=random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )


def _fit_binary(
    X: np.ndarray,
    y_str: np.ndarray,
    class_names: list[str],
    random_state: int = 42,
    class_weight_multipliers: dict | None = None,
):
    class_to_idx, _ = build_class_maps(class_names)
    model = _make_binary_xgb(random_state)
    y_idx = encode_labels(y_str, class_to_idx)
    sample_weight = _sample_weights(y_str, class_weight_multipliers)
    model.fit(X, y_idx, sample_weight=sample_weight)
    return model


def combine_stage_proba(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """P(real), P(diffusion), P(deepfake) = CLASS_NAMES 순서."""
    p_real = p1[:, 0]
    p_fake = p1[:, 1]
    p_diffusion = p_fake * p2[:, 0]
    p_deepfake = p_fake * p2[:, 1]
    return np.column_stack([p_real, p_diffusion, p_deepfake])


def cascade_from_proba(p1: np.ndarray, p2: np.ndarray, fake_threshold: float = 0.5):
    _, stage2_i2c = build_class_maps(STAGE2_NAMES)
    pred1 = np.where(p1[:, 1] >= fake_threshold, "fake", "real")
    pred2 = decode_labels(np.argmax(p2, axis=1), stage2_i2c)
    final = pred1.astype(object).copy()
    fake_mask = pred1 == "fake"
    final[fake_mask] = pred2[fake_mask]
    proba_final = combine_stage_proba(p1, p2)
    return pred1, pred2, final, proba_final


def predict_hierarchical(
    model1,
    model2,
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    fake_threshold: float = 0.5,
):
    if X2 is None:
        X2 = X1
    p1 = model1.predict_proba(X1)
    p2 = model2.predict_proba(X2)
    pred1, pred2, final, proba_final = cascade_from_proba(p1, p2, fake_threshold)
    return pred1, pred2, final, p1, p2, proba_final


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray,
    labels: list[str],
    pos_label: str,
) -> dict:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    try:
        auc = roc_auc_score((np.asarray(y_true) == pos_label).astype(int), proba_pos)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "macro_f1": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "weighted_f1": f1_w,
        "roc_auc": auc,
    }


def binary_metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Accuracy", metrics["accuracy"]),
            ("Macro Precision", metrics["precision_macro"]),
            ("Macro Recall", metrics["recall_macro"]),
            ("Macro F1", metrics["macro_f1"]),
            ("Weighted Precision", metrics["precision_weighted"]),
            ("Weighted Recall", metrics["recall_weighted"]),
            ("Weighted F1", metrics["weighted_f1"]),
            ("ROC AUC", metrics["roc_auc"]),
        ],
        columns=["Metric", "Score"],
    )


def binary_per_class_table(y_true, y_pred, labels, display) -> pd.DataFrame:
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = [
        {
            "class": display[c],
            "precision": p[i],
            "recall": r[i],
            "f1_score": f1[i],
            "support": int(sup[i]),
        }
        for i, c in enumerate(labels)
    ]
    for name, avg in (("Macro Avg", "macro"), ("Weighted Avg", "weighted")):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        rows.append(
            {
                "class": name,
                "precision": pa,
                "recall": ra,
                "f1_score": fa,
                "support": int(len(y_true)),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true, y_pred, labels, display, out_png: Path, title: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [display[c] for c in labels]
    cm_df = pd.DataFrame(cm, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black", fontsize=13,
            )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return cm_df


def error_source(true_label: str, pred_stage1: str, pred_final: str) -> str:
    if true_label == pred_final:
        return "correct"
    if true_label == "real":
        return "stage1"
    if pred_stage1 == "real":
        return "stage1"
    return "stage2"


def evaluate_hierarchical_cv(
    X1: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    class_weight_multipliers: dict | None = None,
    fake_threshold: float = 0.5,
    verbose: bool = True,
    X2: np.ndarray | None = None,
):
    if X2 is None:
        X2 = X1
    y_bin = to_binary_labels(y)
    n = len(y)
    pred1_all = np.empty(n, dtype=object)
    pred2_all = np.empty(n, dtype=object)
    final_all = np.empty(n, dtype=object)
    p1_all = np.zeros((n, 2))
    p2_all = np.zeros((n, 2))
    proba_final_all = np.zeros((n, 3))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X1, y), 1):
        model1 = _fit_binary(
            X1[train_idx], y_bin[train_idx], STAGE1_NAMES,
            random_state, class_weight_multipliers,
        )
        fake_train = y_bin[train_idx] == "fake"
        model2 = _fit_binary(
            X2[train_idx][fake_train], y[train_idx][fake_train], STAGE2_NAMES,
            random_state, class_weight_multipliers,
        )
        pred1, pred2, final, p1, p2, proba_final = predict_hierarchical(
            model1, model2, X1[val_idx], X2[val_idx], fake_threshold,
        )
        pred1_all[val_idx] = pred1
        pred2_all[val_idx] = pred2
        final_all[val_idx] = final
        p1_all[val_idx] = p1
        p2_all[val_idx] = p2
        proba_final_all[val_idx] = proba_final

        if verbose:
            true_fake = y_bin[val_idx] == "fake"
            stage1_acc = accuracy_score(y_bin[val_idx], pred1)
            stage2_acc = (
                accuracy_score(y[val_idx][true_fake], pred2[true_fake])
                if true_fake.any() else float("nan")
            )
            final_acc = accuracy_score(y[val_idx], final)
            print(
                f"  fold {fold}: stage1 acc={stage1_acc:.3f} | "
                f"stage2 oracle acc={stage2_acc:.3f} | final acc={final_acc:.3f}"
            )

    stage1_metrics = compute_binary_metrics(
        y_bin, pred1_all, p1_all[:, 1], STAGE1_NAMES, pos_label="fake"
    )
    true_fake = y_bin == "fake"
    stage2_oracle_metrics = compute_binary_metrics(
        y[true_fake], pred2_all[true_fake], p2_all[true_fake, 1],
        STAGE2_NAMES, pos_label="deepfake",
    )
    final_metrics = compute_metrics(y, final_all, proba_final_all)

    caught_fake = true_fake & (pred1_all == "fake")
    if caught_fake.any():
        stage2_given_stage1 = compute_binary_metrics(
            y[caught_fake], pred2_all[caught_fake], p2_all[caught_fake, 1],
            STAGE2_NAMES, pos_label="deepfake",
        )
    else:
        stage2_given_stage1 = None

    if verbose:
        print("\n=== train 내부 5-fold: 1단계 real vs fake ===")
        print(binary_metrics_table(stage1_metrics).round(4).to_string(index=False))
        print(classification_report(
            y_bin, pred1_all, labels=STAGE1_ORDER,
            target_names=[STAGE1_DISPLAY[c] for c in STAGE1_ORDER],
            digits=3, zero_division=0,
        ))
        print("=== train 내부 5-fold: 2단계 oracle (정답 fake만) ===")
        print(binary_metrics_table(stage2_oracle_metrics).round(4).to_string(index=False))
        print(classification_report(
            y[true_fake], pred2_all[true_fake], labels=STAGE2_ORDER,
            target_names=[STAGE2_DISPLAY[c] for c in STAGE2_ORDER],
            digits=3, zero_division=0,
        ))
        print("=== train 내부 5-fold: 최종 3진 (1단계 오류 포함) ===")
        print(metrics_table(final_metrics).round(4).to_string(index=False))
        print(classification_report(
            y, final_all, labels=DISPLAY_ORDER,
            target_names=[DISPLAY_LABEL[c] for c in DISPLAY_ORDER],
            digits=3, zero_division=0,
        ))

    return {
        "pred_stage1": pred1_all,
        "pred_stage2": pred2_all,
        "pred_final": final_all,
        "proba_stage1": p1_all,
        "proba_stage2": p2_all,
        "proba_final": proba_final_all,
        "stage1_metrics": stage1_metrics,
        "stage2_oracle_metrics": stage2_oracle_metrics,
        "stage2_given_stage1_metrics": stage2_given_stage1,
        "final_metrics": final_metrics,
    }


def _fake_recall(y_true_3class: np.ndarray, pred_stage1: np.ndarray) -> float:
    y_bin = to_binary_labels(y_true_3class)
    fake = y_bin == "fake"
    if not fake.any():
        return float("nan")
    return float((pred_stage1[fake] == "fake").mean())


def score_threshold(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, fake_threshold: float) -> dict:
    pred1, _, final, proba_final = cascade_from_proba(p1, p2, fake_threshold)
    metrics = compute_metrics(y, final, proba_final)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_auc": metrics["macro_auc"],
        "fake_recall": _fake_recall(y, pred1),
        "stage1_acc": accuracy_score(to_binary_labels(y), pred1),
    }


def run_tune(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int,
    random_state: int,
    class_weight_multipliers: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """train CV로 피처 프리셋 × fake 가중치 × 임계값을 고른다. test는 보지 않는다."""
    rows = []
    extra_weight = (class_weight_multipliers or {}).get("fake")
    fake_weights = list(TUNE_FAKE_WEIGHTS)
    if extra_weight is not None and extra_weight not in fake_weights:
        fake_weights.append(float(extra_weight))

    print("\n=== train CV 튜닝 (피처 프리셋 × fake 가중치 × 임계값) ===")
    print("기준: 최종 3진 Macro F1 (동점이면 Accuracy). test set은 사용하지 않는다.")

    for preset_name, (feats1, feats2) in feature_presets().items():
        X1 = slice_features(df_train, feats1)
        X2 = slice_features(df_train, feats2)
        for fake_w in fake_weights:
            weights = dict(class_weight_multipliers or {})
            weights["fake"] = fake_w
            cv = evaluate_hierarchical_cv(
                X1, y_train,
                n_splits=n_splits,
                random_state=random_state,
                class_weight_multipliers=weights,
                fake_threshold=0.5,
                verbose=False,
                X2=X2,
            )
            for thr in TUNE_THRESHOLDS:
                scored = score_threshold(
                    y_train, cv["proba_stage1"], cv["proba_stage2"], thr,
                )
                rows.append(
                    {
                        "preset": preset_name,
                        "fake_weight": fake_w,
                        "threshold": thr,
                        "cv_macro_f1": scored["macro_f1"],
                        "cv_accuracy": scored["accuracy"],
                        "cv_macro_auc": scored["macro_auc"],
                        "cv_fake_recall": scored["fake_recall"],
                        "cv_stage1_acc": scored["stage1_acc"],
                        "n_stage1": len(feats1),
                        "n_stage2": len(feats2),
                        "stage1_features": ",".join(feats1),
                        "stage2_features": ",".join(feats2),
                    }
                )
            base = score_threshold(y_train, cv["proba_stage1"], cv["proba_stage2"], 0.5)
            print(
                f"  preset={preset_name:<18s} fake_w={fake_w:.1f} | "
                f"thr=0.50 macro_f1={base['macro_f1']:.4f}"
            )

    result = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1", "cv_accuracy", "cv_fake_recall"],
        ascending=False,
    ).reset_index(drop=True)
    best = result.iloc[0].to_dict()
    print("\n=== 튜닝 상위 15개 ===")
    print(
        result.head(15)[
            ["preset", "fake_weight", "threshold", "cv_macro_f1", "cv_accuracy", "cv_fake_recall"]
        ].round(4).to_string(index=False)
    )
    print(
        f"\n[선택] preset={best['preset']}, fake_weight={best['fake_weight']}, "
        f"threshold={best['threshold']:.2f} | CV Macro F1={best['cv_macro_f1']:.4f}, "
        f"Acc={best['cv_accuracy']:.4f}"
    )
    return result, best


def run_diagnosis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    pred_stage1: np.ndarray,
    pred_stage2: np.ndarray,
    pred_final: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    proba_final: np.ndarray,
    out_csv: Path,
):
    result = df[["video_name"]].copy()
    result["subgroup"] = result["video_name"].apply(classify_subgroup)
    result["true_label"] = y_true
    result["pred_stage1"] = pred_stage1
    result["pred_stage2"] = pred_stage2
    result["predicted"] = pred_final
    result["correct"] = result["true_label"] == result["predicted"]
    result["error_source"] = [
        error_source(t, s1, f)
        for t, s1, f in zip(y_true, pred_stage1, pred_final)
    ]
    result["prob_real"] = p1[:, 0]
    result["prob_fake"] = p1[:, 1]
    result["prob_diffusion_given_fake"] = p2[:, 0]
    result["prob_deepfake_given_fake"] = p2[:, 1]
    for i, c in enumerate(CLASS_NAMES):
        result[f"prob_{c}"] = proba_final[:, i]
    result["confidence"] = proba_final.max(axis=1)

    print("\n=== 세부 그룹(생성모델/기법)별 최종 정확도 ===")
    summary = (
        result.groupby("subgroup")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("accuracy")
    )
    print(summary.round(3).to_string())

    print("\n=== 오분류 원인 (stage1=진위 실패, stage2=종류 실패) ===")
    src = result["error_source"].value_counts()
    print(src.to_string())

    wrong = result[~result["correct"]].sort_values("confidence", ascending=False)
    wrong.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 오분류 영상 {len(wrong)}개 -> {out_csv}")
    print("\n=== 가장 '확신하며' 틀린 영상 top 10 ===")
    cols = ["video_name", "subgroup", "true_label", "pred_stage1", "predicted", "error_source", "confidence"]
    print(wrong.head(10)[cols].to_string(index=False))
    return result


def _binary_shap_values(model, X: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def run_binary_shap(model, X: np.ndarray, features: list[str], out_dir: Path, tag: str, title: str):
    import shap

    sv = _binary_shap_values(model, X)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    importance = pd.DataFrame(
        {
            "rank": range(1, len(features) + 1),
            "feature": [features[i] for i in order],
            "mean_abs_shap": [mean_abs[i] for i in order],
        }
    )
    print(f"\n=== SHAP Feature Importance ({tag}) ===")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<28s} {row['mean_abs_shap']:.4f}")

    importance.to_csv(out_dir / f"shap_feature_importance_{tag}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(features) + 2))
    ax.barh([features[i] for i in order][::-1], [mean_abs[i] for i in order][::-1], color="#4c72b0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"shap_bar_{tag}.png", dpi=150)
    plt.close(fig)

    fig = plt.figure()
    shap.summary_plot(
        sv, features=X, feature_names=features, show=False,
        plot_size=(7, 0.45 * len(features) + 2),
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{tag}.png", dpi=150)
    plt.close(fig)
    return importance


def explain_single_video(
    model1,
    model2,
    x1: np.ndarray,
    x2: np.ndarray,
    feat1: list[str],
    feat2: list[str],
    fake_threshold: float,
):
    import shap

    pred1, pred2, final, p1, p2, proba_final = predict_hierarchical(
        model1, model2, x1.reshape(1, -1), x2.reshape(1, -1), fake_threshold,
    )
    pred_class = str(final[0])
    print(f"\n예측: {pred_class} (결합 확률 {proba_final[0, CLASS_NAMES.index(pred_class)]*100:.1f}%)")
    print(f"  [1단계] real {p1[0, 0]*100:.1f}% vs fake {p1[0, 1]*100:.1f}%  -> {pred1[0]}")
    print(
        f"  [2단계 | fake] diffusion {p2[0, 0]*100:.1f}% vs deepfake {p2[0, 1]*100:.1f}%"
        + ("  (적용됨)" if pred1[0] == "fake" else "  (1단계가 real이라 미적용)")
    )
    for c in DISPLAY_ORDER:
        print(f"  - 최종 P({c}): {proba_final[0, CLASS_NAMES.index(c)]*100:.1f}%")

    if pred1[0] == "real":
        model, x_row, feature_names = model1, x1, feat1
        tag = "1단계(real vs fake), fake 쪽 기여"
    else:
        model, x_row, feature_names = model2, x2, feat2
        tag = "2단계(deepfake vs diffusion), deepfake 쪽 기여"

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x_row.reshape(1, -1))
    if isinstance(sv, list):
        contrib = np.asarray(sv[1][0])
    else:
        sv = np.asarray(sv)
        contrib = sv[0, :, -1] if sv.ndim == 3 else sv[0]
    order = np.argsort(np.abs(contrib))[::-1]
    print(f"\n{tag}:")
    for i in order:
        sign = "+" if contrib[i] > 0 else "-"
        print(f"  {sign} {feature_names[i]:<25s} 값={x_row[i]:.4f}   기여도={contrib[i]:+.4f}")


def load_compare_metrics(compare_dir: Path | None) -> pd.DataFrame | None:
    if compare_dir is None:
        return None
    path = compare_dir / "final_metrics.csv"
    if not path.is_file():
        print(f"[경고] 3진 비교 파일을 찾지 못했습니다: {path}")
        return None
    return pd.read_csv(path)


def build_comparison_table(hier_metrics: dict, three_class: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Accuracy": "accuracy",
        "Macro Precision": "precision_macro",
        "Macro Recall": "recall_macro",
        "Macro F1": "macro_f1",
        "Weighted F1": "weighted_f1",
        "Macro AUC": "macro_auc",
    }
    three = {str(r["Metric"]): float(r["Score"]) for _, r in three_class.iterrows()}
    rows = []
    for name, key in mapping.items():
        if name not in three or key not in hier_metrics:
            continue
        h = float(hier_metrics[key])
        t = three[name]
        rows.append(
            {
                "Metric": name,
                "3-class": t,
                "Hierarchical": h,
                "Delta (hier - 3class)": h - t,
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(ctx: dict) -> str:
    parts = []
    a = parts.append
    a("# train_classifier2 Hierarchical Experiment Results\n")
    a(f"생성 시각: {ctx['timestamp']}\n")
    a("## 1. Experiment Setup\n")
    a("- Task: Hierarchical 2-stage classification")
    a("  - Stage 1: Real vs Fake")
    a("  - Stage 2: Deepfake vs Diffusion (only if Stage 1 predicts fake)")
    a(f"- Dataset: `{ctx['csv_path']}`")
    a("- Model: XGBoost binary:logistic x 2 (n_estimators=300, max_depth=4, lr=0.05)")
    a(f"- Stage 1 features ({len(ctx['stage1_features'])}개): {', '.join(ctx['stage1_features'])}")
    a(f"- Stage 2 features ({len(ctx['stage2_features'])}개): {', '.join(ctx['stage2_features'])}")
    a("- Labels / train-test split: `train_classifier2.py` 와 동일")
    a(f"- Fake threshold: {ctx['fake_threshold']}")
    a("- Class Weight: balanced sample weight"
      + (f" + {ctx['class_weight']}" if ctx["class_weight"] else ""))
    a(f"- Validation: train 내부 {ctx['n_splits']}-Fold Stratified CV")
    a("- Test Set: 최종 평가에만 1회 사용")
    a(f"- Total Samples: {ctx['n_total']}")
    a(f"- Train / Test: {ctx['n_train']} / {ctx['n_test']}\n")
    a("클래스별 샘플 수\n")
    a(_md_table(ctx["class_counts"], float_fmt="{:.0f}"))
    a("")
    a("## 2. Features\n")
    a("### Stage 1\n")
    a(_bullets(ctx["stage1_features"]))
    a("\n### Stage 2\n")
    a(_bullets(ctx["stage2_features"]))
    a("")
    a("## 3. Train CV Performance\n")
    a("### 3.1 Stage 1 (real vs fake)\n")
    a(_md_table(ctx["cv_stage1_table"]))
    a("")
    a("### 3.2 Stage 2 oracle (정답 fake만, 1단계 무시)\n")
    a(_md_table(ctx["cv_stage2_table"]))
    a("")
    a("### 3.3 End-to-end 3-class\n")
    a(_md_table(ctx["cv_final_table"]))
    a("")
    a("## 4. Final Model Performance (test set)\n")
    a("### 4.1 Stage 1 (real vs fake)\n")
    a(_md_table(ctx["test_stage1_table"]))
    a("")
    a(_md_table(ctx["test_stage1_per_class"]))
    a("")
    a("```text")
    a(ctx["stage1_report"].rstrip())
    a("```\n")
    a("### 4.2 Stage 2 oracle (정답 fake만)\n")
    a(_md_table(ctx["test_stage2_table"]))
    a("")
    a(_md_table(ctx["test_stage2_per_class"]))
    a("")
    a("```text")
    a(ctx["stage2_report"].rstrip())
    a("```\n")
    if ctx.get("test_stage2_given_table") is not None:
        a("### 4.3 Stage 2 given Stage 1 (1단계가 fake로 맞춘 정답 fake만)\n")
        a(_md_table(ctx["test_stage2_given_table"]))
        a("")
    a("### 4.4 End-to-end 3-class (1단계 오류 포함)\n")
    a(_md_table(ctx["test_final_table"]))
    a("")
    a(_md_table(ctx["per_class"]))
    a("")
    a("```text")
    a(ctx["clf_report"].rstrip())
    a("```\n")
    a("## 5. Confusion Matrices\n")
    a("### Stage 1\n")
    cm1 = ctx["cm_stage1"].copy()
    cm1.insert(0, "Actual \\ Predicted", cm1.index)
    a(_md_table(cm1, float_fmt="{:.0f}"))
    a("")
    a("![Stage 1 Confusion Matrix](confusion_matrix_stage1.png)\n")
    a("### Stage 2 oracle\n")
    cm2 = ctx["cm_stage2"].copy()
    cm2.insert(0, "Actual \\ Predicted", cm2.index)
    a(_md_table(cm2, float_fmt="{:.0f}"))
    a("")
    a("![Stage 2 Confusion Matrix](confusion_matrix_stage2_oracle.png)\n")
    a("### End-to-end 3-class\n")
    cmf = ctx["cm_final"].copy()
    cmf.insert(0, "Actual \\ Predicted", cmf.index)
    a(_md_table(cmf, float_fmt="{:.0f}"))
    a("")
    a("![Final Confusion Matrix](confusion_matrix.png)\n")
    a("## 6. Error Source\n")
    a(_md_table(ctx["error_source_table"], float_fmt="{:.0f}"))
    a("")
    a("## 7. SHAP Feature Importance\n")
    a("### Stage 1 (real vs fake, test set)\n")
    a(_md_table(ctx["shap_stage1"]))
    a("")
    a("![SHAP Stage 1](shap_summary_stage1.png)\n")
    a("### Stage 2 (deepfake vs diffusion, true fake test samples)\n")
    a(_md_table(ctx["shap_stage2"]))
    a("")
    a("![SHAP Stage 2](shap_summary_stage2.png)\n")
    if ctx.get("comparison") is not None:
        a("## 8. Comparison with 3-class model\n")
        a(f"비교 폴더: `{ctx['compare_dir']}`\n")
        a(_md_table(ctx["comparison"]))
        a("")
        a("## 9. Summary\n")
    else:
        a("## 8. Summary\n")
    a(ctx["summary_text"])
    a("")
    return "\n".join(parts)


def build_summary_text(ctx: dict) -> str:
    s1 = ctx["test_stage1_metrics"]
    s2 = ctx["test_stage2_metrics"]
    m = ctx["test_final_metrics"]
    err = ctx["error_source_table"]
    n_s1 = int(err.loc[err["source"] == "stage1", "count"].sum()) if (err["source"] == "stage1").any() else 0
    n_s2 = int(err.loc[err["source"] == "stage2", "count"].sum()) if (err["source"] == "stage2").any() else 0
    lines = [
        f"- 1단계 feature {len(ctx['stage1_features'])}개, 2단계 feature {len(ctx['stage2_features'])}개, "
        f"P(fake) 임계값 {ctx['fake_threshold']} 로 train {ctx['n_train']} / test {ctx['n_test']} 를 사용했다.",
        f"- 1단계(real vs fake) test Accuracy {s1['accuracy']:.4f}, Macro F1 {s1['macro_f1']:.4f}, "
        f"ROC AUC {s1['roc_auc']:.4f}.",
        f"- 2단계 oracle(정답 fake만) test Accuracy {s2['accuracy']:.4f}, Macro F1 {s2['macro_f1']:.4f}, "
        f"ROC AUC {s2['roc_auc']:.4f}.",
        f"- 최종 3진 test Accuracy {m['accuracy']:.4f}, Macro F1 {m['macro_f1']:.4f}, "
        f"Macro AUC {m['macro_auc']:.4f}.",
        f"- test 오분류 중 1단계(진위) 실패 {n_s1}개, 2단계(종류) 실패 {n_s2}개.",
    ]
    top1 = ctx["shap_stage1"].iloc[0]
    top2 = ctx["shap_stage2"].iloc[0]
    lines.append(
        f"- SHAP 1단계 최상위 feature 는 `{top1['feature']}` "
        f"(mean |SHAP| = {top1['mean_abs_shap']:.4f}), "
        f"2단계는 `{top2['feature']}` (mean |SHAP| = {top2['mean_abs_shap']:.4f}) 였다."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="1단계 real-vs-fake → 2단계 deepfake-vs-diffusion 계층적 분류"
    )
    parser.add_argument("--csv", required=True, help="입력 CSV 경로 (unified_features_*.csv)")
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR),
                         help="진짜 영상 폴더 (여기 있는 파일명만 real로 라벨링)")
    parser.add_argument("--test-size", type=float, default=0.2, help="최종 평가용 test 비율 (기본 0.2)")
    parser.add_argument("--n-splits", type=int, default=5, help="train 내부 교차검증 fold 수 (기본 5)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fake-threshold", type=float, default=0.45,
                         help="1단계에서 fake 로 보낼 P(fake) 임계값 (기본 0.45, train CV 튜닝 결과)")
    parser.add_argument(
        "--stage1-features",
        default="identity_sim_std,bvp_std,highfreq_score,boundary_score_mean,boundary_score_std",
        help="1단계 피처. 'all' 또는 콤마 구분 이름",
    )
    parser.add_argument(
        "--stage2-features",
        default="bvp_std,identity_sim_std,boundary_score_std,boundary_score_mean,highfreq_score",
        help="2단계 피처. 'all' 또는 콤마 구분 이름",
    )
    parser.add_argument("--tune", action="store_true",
                         help="train CV로 피처 프리셋/fake 가중치/임계값을 고른 뒤 그 설정으로 학습")
    parser.add_argument("--class-weight", default=None,
                         help="클래스별 가중치 배수. 예: 'fake=1.2' 또는 'diffusion=3.0,deepfake=1.0'")
    parser.add_argument("--diagnose", action="store_true",
                         help="오분류된 영상을 찾아 CSV로 저장하고, 생성모델/기법별 오류율을 출력")
    parser.add_argument("--remove-outliers", action="store_true",
                         help="IQR 기반 이상치 제거 적용 (기본: 미적용)")
    parser.add_argument("--compare-dir", default=None,
                         help="기존 3진 결과 폴더 (final_metrics.csv 가 있으면 비교표 추가)")
    parser.add_argument("--model-out", default=None, help="최종 모델 저장 경로 (기본: 결과 폴더 안)")
    parser.add_argument("--explain-n", type=int, default=3, help="test 영상 중 예시로 설명을 출력할 개수")
    parser.add_argument("--out-root", default="results", help="결과 디렉터리의 상위 경로")
    parser.add_argument("--log-out", default=None, help="학습 로그 txt 경로 (기본: 결과 폴더의 run_log.txt)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"train_classifier2_hierarchical_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_out) if args.log_out else out_dir / "run_log.txt"

    tee = Tee(sys.stdout, log_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        _run(args, out_dir, stamp)
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"\nResults saved to:\n{out_dir.as_posix()}/")


def _run(args, out_dir: Path, stamp: str):
    print(f"[로드] {args.csv}")
    df = load_and_label(args.csv, real_dir=args.real_dir)
    print(df.groupby("group").size())

    if args.remove_outliers:
        df = remove_outliers_iqr(df, FEATURE_COLS)

    y_all = df["group"].values
    class_weight_multipliers = {}
    if args.class_weight:
        class_weight_multipliers = parse_class_weight(args.class_weight)
        print(f"[클래스 가중치 배수 적용] {class_weight_multipliers}")

    df_train, df_test, y_train, y_test = train_test_split(
        df, y_all, test_size=args.test_size, stratify=y_all, random_state=args.random_state,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train_bin = to_binary_labels(y_train)
    y_test_bin = to_binary_labels(y_test)

    print(f"\n[데이터 분리] 전체 {len(df)}개 -> train {len(df_train)} / test {len(df_test)}")
    print("train 3진:", pd.Series(y_train).value_counts().to_dict())
    print("train 1단계:", pd.Series(y_train_bin).value_counts().to_dict())
    print("test  3진:", pd.Series(y_test).value_counts().to_dict())
    print("test  1단계:", pd.Series(y_test_bin).value_counts().to_dict())

    stage1_feats = parse_features(args.stage1_features)
    stage2_feats = parse_features(args.stage2_features)
    fake_threshold = args.fake_threshold
    tune_table = None
    chosen_preset = "custom"
    if (
        stage1_feats == feature_presets()["top_split"][0]
        and stage2_feats == feature_presets()["top_split"][1]
        and abs(fake_threshold - 0.45) < 1e-9
    ):
        chosen_preset = "top_split"

    if args.tune:
        tune_table, best = run_tune(
            df_train, y_train, args.n_splits, args.random_state, class_weight_multipliers,
        )
        tune_table.to_csv(out_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")
        stage1_feats = best["stage1_features"].split(",")
        stage2_feats = best["stage2_features"].split(",")
        fake_threshold = float(best["threshold"])
        class_weight_multipliers = dict(class_weight_multipliers)
        class_weight_multipliers["fake"] = float(best["fake_weight"])
        chosen_preset = best["preset"]
        print(f"[튜닝 결과 저장] {out_dir / 'tuning_results.csv'}")

    print("\n[계층적 분류] 1단계 real vs fake → 2단계 deepfake vs diffusion")
    print(f"[프리셋] {chosen_preset}")
    print(f"[1단계 피처] {stage1_feats}")
    print(f"[2단계 피처] {stage2_feats}")
    print(f"[1단계 임계값] P(fake) >= {fake_threshold} 이면 2단계로 전달")
    if class_weight_multipliers:
        print(f"[클래스 가중치 배수] {class_weight_multipliers}")

    X1_train = slice_features(df_train, stage1_feats)
    X2_train = slice_features(df_train, stage2_feats)
    X1_test = slice_features(df_test, stage1_feats)
    X2_test = slice_features(df_test, stage2_feats)

    print("\n=== train 내부 5-fold 교차검증 ===")
    cv = evaluate_hierarchical_cv(
        X1_train, y_train,
        n_splits=args.n_splits,
        random_state=args.random_state,
        class_weight_multipliers=class_weight_multipliers,
        fake_threshold=fake_threshold,
        X2=X2_train,
    )

    print("\n=== train 데이터로 최종 1·2단계 모델 학습 ===")
    model1 = _fit_binary(
        X1_train, y_train_bin, STAGE1_NAMES,
        args.random_state, class_weight_multipliers,
    )
    fake_train = y_train_bin == "fake"
    model2 = _fit_binary(
        X2_train[fake_train], y_train[fake_train], STAGE2_NAMES,
        args.random_state, class_weight_multipliers,
    )
    model_out = args.model_out or str(out_dir / "xgb_model_hierarchical.joblib")
    joblib.dump(
        {
            "stage1_model": model1,
            "stage2_model": model2,
            "feature_cols": FEATURE_COLS,
            "stage1_features": stage1_feats,
            "stage2_features": stage2_feats,
            "stage1_classes": STAGE1_NAMES,
            "stage2_classes": STAGE2_NAMES,
            "class_names": CLASS_NAMES,
            "fake_threshold": fake_threshold,
        },
        model_out,
    )
    print(f"[저장 완료] 계층적 최종 모델 -> {model_out}")

    pred1, pred2, y_pred, p1, p2, proba_final = predict_hierarchical(
        model1, model2, X1_test, X2_test, fake_threshold,
    )

    test_stage1_metrics = compute_binary_metrics(
        y_test_bin, pred1, p1[:, 1], STAGE1_NAMES, pos_label="fake"
    )
    true_fake_test = y_test_bin == "fake"
    test_stage2_metrics = compute_binary_metrics(
        y_test[true_fake_test], pred2[true_fake_test], p2[true_fake_test, 1],
        STAGE2_NAMES, pos_label="deepfake",
    )
    caught_fake = true_fake_test & (pred1 == "fake")
    test_stage2_given = None
    test_stage2_given_table = None
    if caught_fake.any():
        test_stage2_given = compute_binary_metrics(
            y_test[caught_fake], pred2[caught_fake], p2[caught_fake, 1],
            STAGE2_NAMES, pos_label="deepfake",
        )
        test_stage2_given_table = binary_metrics_table(test_stage2_given)
    test_final_metrics = compute_metrics(y_test, y_pred, proba_final)

    test_stage1_table = binary_metrics_table(test_stage1_metrics)
    test_stage2_table = binary_metrics_table(test_stage2_metrics)
    test_final_table = metrics_table(test_final_metrics)
    cv_stage1_table = binary_metrics_table(cv["stage1_metrics"])
    cv_stage2_table = binary_metrics_table(cv["stage2_oracle_metrics"])
    cv_final_table = metrics_table(cv["final_metrics"])

    stage1_report = classification_report(
        y_test_bin, pred1, labels=STAGE1_ORDER,
        target_names=[STAGE1_DISPLAY[c] for c in STAGE1_ORDER],
        digits=3, zero_division=0,
    )
    stage2_report = classification_report(
        y_test[true_fake_test], pred2[true_fake_test], labels=STAGE2_ORDER,
        target_names=[STAGE2_DISPLAY[c] for c in STAGE2_ORDER],
        digits=3, zero_division=0,
    )
    clf_report = classification_report(
        y_test, y_pred, labels=DISPLAY_ORDER,
        target_names=[DISPLAY_LABEL[c] for c in DISPLAY_ORDER],
        digits=3, zero_division=0,
    )
    per_class = per_class_table(y_test, y_pred)
    stage1_per_class = binary_per_class_table(y_test_bin, pred1, STAGE1_ORDER, STAGE1_DISPLAY)
    stage2_per_class = binary_per_class_table(
        y_test[true_fake_test], pred2[true_fake_test], STAGE2_ORDER, STAGE2_DISPLAY
    )

    print("\n=== test set: 1단계 real vs fake ===")
    print(test_stage1_table.round(4).to_string(index=False))
    print(stage1_report)
    print("=== test set: 2단계 oracle (정답 fake만) ===")
    print(test_stage2_table.round(4).to_string(index=False))
    print(stage2_report)
    if test_stage2_given is not None:
        print("=== test set: 2단계 given 1단계 (1단계가 fake로 맞춘 정답 fake만) ===")
        print(test_stage2_given_table.round(4).to_string(index=False))
    print("=== test set: 최종 3진 ===")
    print(test_final_table.round(4).to_string(index=False))
    print(clf_report)

    cm_stage1 = save_confusion_matrix(
        y_test_bin, pred1, STAGE1_ORDER, STAGE1_DISPLAY,
        out_dir / "confusion_matrix_stage1.png", "Stage 1: Real vs Fake (test)",
    )
    cm_stage2 = save_confusion_matrix(
        y_test[true_fake_test], pred2[true_fake_test], STAGE2_ORDER, STAGE2_DISPLAY,
        out_dir / "confusion_matrix_stage2_oracle.png",
        "Stage 2 oracle: Deepfake vs Diffusion (true fake test)",
    )
    cm_final = save_confusion_matrix(
        y_test, y_pred, DISPLAY_ORDER, DISPLAY_LABEL,
        out_dir / "confusion_matrix.png", "End-to-end 3-class (test)",
    )
    print("=== Confusion Matrix 1단계 ===")
    print(cm_stage1.to_string())
    print("=== Confusion Matrix 2단계 oracle ===")
    print(cm_stage2.to_string())
    print("=== Confusion Matrix 최종 3진 ===")
    print(cm_final.to_string())

    sources = np.array(
        [error_source(t, s1, f) for t, s1, f in zip(y_test, pred1, y_pred)],
        dtype=object,
    )
    error_source_table = (
        pd.Series(sources, name="count")
        .value_counts()
        .rename_axis("source")
        .reset_index()
    )
    print("\n=== test 오분류 원인 ===")
    print(error_source_table.to_string(index=False))

    cv_stage1_table.to_csv(out_dir / "cv_stage1_metrics.csv", index=False, encoding="utf-8-sig")
    cv_stage2_table.to_csv(out_dir / "cv_stage2_oracle_metrics.csv", index=False, encoding="utf-8-sig")
    cv_final_table.to_csv(out_dir / "cv_final_metrics.csv", index=False, encoding="utf-8-sig")
    test_stage1_table.to_csv(out_dir / "stage1_metrics.csv", index=False, encoding="utf-8-sig")
    test_stage2_table.to_csv(out_dir / "stage2_oracle_metrics.csv", index=False, encoding="utf-8-sig")
    test_final_table.to_csv(out_dir / "final_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    stage1_per_class.to_csv(out_dir / "stage1_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    stage2_per_class.to_csv(out_dir / "stage2_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm_stage1.to_csv(out_dir / "confusion_matrix_stage1.csv", encoding="utf-8-sig")
    cm_stage2.to_csv(out_dir / "confusion_matrix_stage2_oracle.csv", encoding="utf-8-sig")
    cm_final.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    error_source_table.to_csv(out_dir / "error_source.csv", index=False, encoding="utf-8-sig")

    pred_df = df_test[["video_name"]].copy()
    pred_df["true_label"] = y_test
    pred_df["true_binary"] = y_test_bin
    pred_df["pred_stage1"] = pred1
    pred_df["pred_stage2"] = pred2
    pred_df["predicted"] = y_pred
    pred_df["error_source"] = sources
    pred_df["prob_real"] = p1[:, 0]
    pred_df["prob_fake"] = p1[:, 1]
    pred_df["prob_diffusion_given_fake"] = p2[:, 0]
    pred_df["prob_deepfake_given_fake"] = p2[:, 1]
    for i, c in enumerate(CLASS_NAMES):
        pred_df[f"prob_{c}"] = proba_final[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    if args.diagnose:
        run_diagnosis(
            df_test, y_test, pred1, pred2, y_pred, p1, p2, proba_final,
            out_dir / "misclassified_videos.csv",
        )

    shap_stage1 = run_binary_shap(
        model1, X1_test, stage1_feats, out_dir, "stage1",
        "SHAP Stage 1: Real vs Fake (test)",
    )
    shap_stage2 = run_binary_shap(
        model2, X2_test[true_fake_test], stage2_feats, out_dir, "stage2",
        "SHAP Stage 2: Deepfake vs Diffusion (true fake test)",
    )

    print(f"\n=== test 예시 영상 {args.explain_n}개에 대한 개별 설명 ===")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X1_test), size=min(args.explain_n, len(X1_test)), replace=False)
    for idx in sample_idx:
        print(f"\n--- {df_test.iloc[idx]['video_name']} (실제 라벨: {y_test[idx]}) ---")
        explain_single_video(
            model1, model2, X1_test[idx], X2_test[idx],
            stage1_feats, stage2_feats, fake_threshold,
        )

    class_counts = pd.DataFrame(
        [
            {
                "Class": DISPLAY_LABEL[c],
                "Total": int((y_all == c).sum()),
                "Train": int((y_train == c).sum()),
                "Test": int((y_test == c).sum()),
            }
            for c in DISPLAY_ORDER
        ]
    )
    compare_dir = Path(args.compare_dir) if args.compare_dir else None
    three_class = load_compare_metrics(compare_dir)
    comparison = build_comparison_table(test_final_metrics, three_class) if three_class is not None else None
    if comparison is not None:
        comparison.to_csv(out_dir / "comparison_vs_3class.csv", index=False, encoding="utf-8-sig")
        print("\n=== 3진 모델과 test 성능 비교 ===")
        print(comparison.round(4).to_string(index=False))

    ctx = {
        "timestamp": datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
        "csv_path": args.csv,
        "n_splits": args.n_splits,
        "fake_threshold": fake_threshold,
        "stage1_features": stage1_feats,
        "stage2_features": stage2_feats,
        "class_weight": class_weight_multipliers or None,
        "n_total": len(df),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "class_counts": class_counts,
        "cv_stage1_table": cv_stage1_table,
        "cv_stage2_table": cv_stage2_table,
        "cv_final_table": cv_final_table,
        "test_stage1_metrics": test_stage1_metrics,
        "test_stage2_metrics": test_stage2_metrics,
        "test_final_metrics": test_final_metrics,
        "test_stage1_table": test_stage1_table,
        "test_stage2_table": test_stage2_table,
        "test_stage2_given_table": test_stage2_given_table,
        "test_final_table": test_final_table,
        "test_stage1_per_class": stage1_per_class,
        "test_stage2_per_class": stage2_per_class,
        "per_class": per_class,
        "stage1_report": stage1_report,
        "stage2_report": stage2_report,
        "clf_report": clf_report,
        "cm_stage1": cm_stage1,
        "cm_stage2": cm_stage2,
        "cm_final": cm_final,
        "error_source_table": error_source_table,
        "shap_stage1": shap_stage1,
        "shap_stage2": shap_stage2,
        "compare_dir": args.compare_dir,
        "comparison": comparison,
    }
    ctx["summary_text"] = build_summary_text(ctx)
    md_path = out_dir / "train_classifier2_hierarchical_results.md"
    md_path.write_text(build_markdown_report(ctx), encoding="utf-8")

    print("\n=== 요약 ===")
    print(ctx["summary_text"])
    print(f"\n[저장 완료] Markdown 보고서 -> {md_path.as_posix()}")


if __name__ == "__main__":
    main()
