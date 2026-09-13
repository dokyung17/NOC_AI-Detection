"""
boundary / identity 피처 선택 실험
================================

분류기 본체가 아니다. train_classifier2.py 와 같은 데이터/모델 설정을 쓰되,
boundary·identity 계열의 조합을 비교해서 어떤 피처가 실제로 도움이 되는지 확인한다.

사용법:
    python select_features.py --csv results/unified_features_20260911_024351.csv
    python select_features.py --csv ... --test-size 0.2 --tol 0.005

진행 순서:
    1. CSV 로드 + train_classifier2 와 동일한 규칙으로 real/diffusion/deepfake 라벨링
    2. 층화 추출로 train / test 분리 (test 는 마지막 최종 평가에만 사용)
    3. train 내부 5-fold 교차검증으로 boundary 피처 부분집합 비교 -> 선택
    4. 같은 방식으로 identity 피처 부분집합 비교 -> 선택
    5. 선택된 피처로 최종 모델 학습 후 test set 평가 (지표 / classification report /
       confusion matrix / SHAP)
    6. 전체 피처 모델과 성능 비교 후 Markdown 보고서 자동 생성

선별 결과를 실제 학습에 쓰려면 train_classifier2.py 의 FEATURE_COLS 를 수정한다.

모든 산출물은 results/select_features_YYYYMMDD_HHMMSS/ 아래에 저장된다.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_classifier2 import (
    CLASS_NAMES,
    DEFAULT_REAL_DIR,
    FEATURE_COLS,
    _make_multiclass_xgb,
    _sample_weights,
    build_class_maps,
    decode_labels,
    encode_labels,
    load_and_label,
    remove_outliers_iqr,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 피처 그룹 정의 (ALL_FEATURES = BASE + BOUNDARY + IDENTITY)
# FEATURE_COLS 는 train_classifier2 에서 이미 선별된 목록을 쓴다.
# ---------------------------------------------------------------------------
ALL_FEATURES = list(FEATURE_COLS)
BOUNDARY_FEATURES = [c for c in ALL_FEATURES if c.startswith("boundary_")]
IDENTITY_FEATURES = [c for c in ALL_FEATURES if c.startswith("identity_")]
BASE_FEATURES = [c for c in ALL_FEATURES if c not in BOUNDARY_FEATURES + IDENTITY_FEATURES]

# 보고서/혼동행렬에 사용할 클래스 표시 순서 (모델 내부 인코딩은 CLASS_NAMES 순서를 따른다)
DISPLAY_ORDER = ["real", "deepfake", "diffusion"]
DISPLAY_LABEL = {"real": "Real", "deepfake": "Deepfake", "diffusion": "Diffusion"}


class Tee:
    """콘솔과 로그 파일에 동시에 출력한다."""

    def __init__(self, stream, log_path: Path):
        self.stream = stream
        self.file = log_path.open("w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# 성능 지표
# ---------------------------------------------------------------------------
def compute_metrics(y_true_str: np.ndarray, y_pred_str: np.ndarray, proba: np.ndarray) -> dict:
    """문자열 라벨 + 확률 행렬(열 순서 = CLASS_NAMES)로부터 모든 지표를 계산한다."""
    class_to_idx, _ = build_class_maps(CLASS_NAMES)
    y_true_idx = encode_labels(y_true_str, class_to_idx)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_str, y_pred_str, labels=CLASS_NAMES, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true_str, y_pred_str, labels=CLASS_NAMES, average="weighted", zero_division=0
    )

    labels_idx = list(range(len(CLASS_NAMES)))
    try:
        auc_macro = roc_auc_score(
            y_true_idx, proba, multi_class="ovr", average="macro", labels=labels_idx
        )
        auc_weighted = roc_auc_score(
            y_true_idx, proba, multi_class="ovr", average="weighted", labels=labels_idx
        )
    except ValueError:
        auc_macro = float("nan")
        auc_weighted = float("nan")

    return {
        "accuracy": accuracy_score(y_true_str, y_pred_str),
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "macro_f1": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "weighted_f1": f1_w,
        "macro_auc": auc_macro,
        "weighted_auc": auc_weighted,
    }


def cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    class_weight_multipliers: dict | None = None,
) -> dict:
    """train 데이터 내부 층화 교차검증 OOF 예측으로 지표를 계산한다."""
    class_to_idx, idx_to_class = build_class_maps(CLASS_NAMES)
    n_classes = len(CLASS_NAMES)
    y_idx = encode_labels(y, class_to_idx)
    sample_weight = _sample_weights(y, class_weight_multipliers)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty_like(y_idx)
    oof_proba = np.zeros((len(y_idx), n_classes))
    for train_idx, val_idx in cv.split(X, y_idx):
        model = _make_multiclass_xgb(n_classes, random_state)
        model.fit(X[train_idx], y_idx[train_idx], sample_weight=sample_weight[train_idx])
        oof_pred[val_idx] = model.predict(X[val_idx])
        oof_proba[val_idx] = model.predict_proba(X[val_idx])

    return compute_metrics(y, decode_labels(oof_pred, idx_to_class), oof_proba)


# ---------------------------------------------------------------------------
# 피처 선택
# ---------------------------------------------------------------------------
def _subset_label(subset: list[str]) -> str:
    return " + ".join(subset) if subset else "(none)"


def select_feature_subset(
    group_name: str,
    fixed_features: list[str],
    candidates: list[str],
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    tol: float,
    n_splits: int,
    random_state: int,
    class_weight_multipliers: dict | None,
) -> tuple[list[str], pd.DataFrame]:
    """
    후보 피처의 모든 부분집합(빈 집합 포함)을 train 내부 CV로 비교한다.
    Macro F1 최고값과의 차이가 tol 이내면 더 적은 피처를 쓰는 조합을 선택한다.
    """
    print(f"\n=== {group_name} 피처 부분집합 비교 (train {len(y_train)}개, {n_splits}-fold CV) ===")
    print(f"후보: {candidates}")

    rows = []
    for size in range(len(candidates) + 1):
        for combo in combinations(candidates, size):
            subset = list(combo)
            feats = fixed_features + subset
            m = cv_metrics(
                df_train[feats].values.astype(float),
                y_train,
                n_splits=n_splits,
                random_state=random_state,
                class_weight_multipliers=class_weight_multipliers,
            )
            rows.append(
                {
                    "feature_group": group_name,
                    "selected_features": _subset_label(subset),
                    "n_group_features": len(subset),
                    "n_total_features": len(feats),
                    "accuracy": m["accuracy"],
                    "precision": m["precision_macro"],
                    "recall": m["recall_macro"],
                    "macro_f1": m["macro_f1"],
                    "macro_auc": m["macro_auc"],
                    "weighted_auc": m["weighted_auc"],
                    "_subset": subset,
                }
            )

    table = pd.DataFrame(rows)
    print(
        table.drop(columns=["_subset", "feature_group"])
        .round(4)
        .to_string(index=False)
    )

    best_f1 = table["macro_f1"].max()
    eligible = table[table["macro_f1"] >= best_f1 - tol].copy()
    eligible = eligible.sort_values(
        ["n_group_features", "macro_f1", "macro_auc"], ascending=[True, False, False]
    )
    chosen = eligible.iloc[0]
    selected = list(chosen["_subset"])
    removed = [c for c in candidates if c not in selected]

    table["is_selected"] = [s == selected for s in table["_subset"]]

    print(f"\n[{group_name}] Macro F1 최고 = {best_f1:.4f} / 허용 오차 tol = {tol}")
    print(
        f"[{group_name}] 선택된 조합: {_subset_label(selected)} "
        f"(Macro F1 = {chosen['macro_f1']:.4f}, Accuracy = {chosen['accuracy']:.4f}, "
        f"Macro AUC = {chosen['macro_auc']:.4f})"
    )
    print(f"=== {group_name} Feature Selection ===")
    print("Selected:")
    for f in selected or ["(없음)"]:
        print(f"- {f}")
    print("Removed:")
    for f in removed or ["(없음)"]:
        print(f"- {f}")

    return selected, table.drop(columns=["_subset"])


# ---------------------------------------------------------------------------
# 최종 학습 / 평가
# ---------------------------------------------------------------------------
def fit_model(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    random_state: int,
    class_weight_multipliers: dict | None,
):
    class_to_idx, _ = build_class_maps(CLASS_NAMES)
    X = df_train[features].values.astype(float)
    y_idx = encode_labels(y_train, class_to_idx)
    model = _make_multiclass_xgb(len(CLASS_NAMES), random_state)
    model.fit(X, y_idx, sample_weight=_sample_weights(y_train, class_weight_multipliers))
    return model


def evaluate_on_test(model, df_test: pd.DataFrame, y_test: np.ndarray, features: list[str]):
    _, idx_to_class = build_class_maps(CLASS_NAMES)
    X = df_test[features].values.astype(float)
    proba = model.predict_proba(X)
    y_pred = decode_labels(model.predict(X), idx_to_class)
    return y_pred, proba, compute_metrics(y_test, y_pred, proba)


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=DISPLAY_ORDER, zero_division=0
    )
    rows = [
        {
            "class": DISPLAY_LABEL[c],
            "precision": p[i],
            "recall": r[i],
            "f1_score": f1[i],
            "support": int(sup[i]),
        }
        for i, c in enumerate(DISPLAY_ORDER)
    ]
    for name, avg in (("Macro Avg", "macro"), ("Weighted Avg", "weighted")):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=DISPLAY_ORDER, average=avg, zero_division=0
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


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=DISPLAY_ORDER)
    display = [DISPLAY_LABEL[c] for c in DISPLAY_ORDER]
    cm_df = pd.DataFrame(cm, index=display, columns=display)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(display)), display)
    ax.set_yticks(range(len(display)), display)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=13,
            )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return cm_df


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def _shap_array(model, X: np.ndarray) -> np.ndarray:
    """(n_samples, n_features, n_classes) 형태로 정규화한 SHAP 값을 돌려준다."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:
        return values[:, :, None]
    return values


def run_shap_analysis(model, df_test: pd.DataFrame, features: list[str], out_dir: Path):
    X = df_test[features].values.astype(float)
    sv = _shap_array(model, X)  # (n, f, c)

    mean_abs = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1]

    importance = pd.DataFrame(
        {
            "rank": range(1, len(features) + 1),
            "feature": [features[i] for i in order],
            "mean_abs_shap": [mean_abs[i] for i in order],
        }
    )
    per_class = pd.DataFrame(
        {"feature": features},
    )
    for ci, cname in enumerate(CLASS_NAMES):
        per_class[f"mean_abs_shap_{cname}"] = np.abs(sv[:, :, ci]).mean(axis=0)
    per_class["mean_abs_shap_overall"] = mean_abs
    per_class = per_class.sort_values("mean_abs_shap_overall", ascending=False)

    print("\n=== SHAP Feature Importance ===")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<28s} {row['mean_abs_shap']:.4f}")

    print("\n=== 클래스별 mean |SHAP| ===")
    print(per_class.round(4).to_string(index=False))

    importance.to_csv(out_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "shap_importance_per_class.csv", index=False, encoding="utf-8-sig")

    # 전체 요약(클래스 평균 기준 bar plot)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(features) + 2))
    ax.barh([features[i] for i in order][::-1], [mean_abs[i] for i in order][::-1], color="#4c72b0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("SHAP Feature Importance (mean over classes, test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close(fig)

    # 클래스별 beeswarm
    import shap

    class_plots = {}
    for ci, cname in enumerate(CLASS_NAMES):
        fig = plt.figure()
        shap.summary_plot(
            sv[:, :, ci],
            features=X,
            feature_names=features,
            show=False,
            plot_size=(7, 0.45 * len(features) + 2),
        )
        plt.title(f"SHAP summary - class: {cname}")
        fname = f"shap_summary_{cname}.png"
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        class_plots[cname] = fname

    return importance, per_class, class_plots


# ---------------------------------------------------------------------------
# Markdown 보고서
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}", right_align_from: int = 1) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    align = "|" + "|".join(
        ("---:" if i >= right_align_from else "---") for i in range(len(df.columns))
    ) + "|"
    lines = [header, align]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if np.isnan(v) else float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _bullets(items) -> str:
    return "\n".join(f"- {i}" for i in items) if len(items) else "- (없음)"


def build_markdown_report(ctx: dict) -> str:
    parts = []
    a = parts.append

    a("# Feature Selection Experiment Results\n")
    a(f"생성 시각: {ctx['timestamp']}\n")

    # 1. 실험 개요
    a("## 1. Experiment Setup\n")
    a(f"- Task: Real / Deepfake / Diffusion 3-class Classification")
    a(f"- Dataset: `{ctx['csv_path']}`")
    a(f"- Model: XGBoost (multi:softprob, n_estimators=300, max_depth=4, lr=0.05)")
    a(f"- Class Weight: balanced sample weight" + (f" + {ctx['class_weight']}" if ctx["class_weight"] else ""))
    a(f"- Feature Selection Metric: Macro F1 (tie-break tolerance = {ctx['tol']}, 동률이면 더 적은 피처 선택)")
    a(f"- Feature Selection Method: boundary/identity 계열 전체 부분집합 완전 탐색 (train 내부 {ctx['n_splits']}-Fold CV)")
    a(f"- Validation Method: {ctx['n_splits']}-Fold Stratified Cross Validation (train set 내부)")
    a(f"- Test Set: 피처 선택 이후 최종 평가에만 1회 사용")
    a(f"- Excluded Features: " + (", ".join(f"`{f}`" for f in ctx["dropped"]) if ctx["dropped"] else "(없음)"))
    a(f"- Total Samples: {ctx['n_total']}")
    a(f"- Train / Test: {ctx['n_train']} / {ctx['n_test']}\n")
    a("클래스별 샘플 수\n")
    a(_md_table(ctx["class_counts"], float_fmt="{:.0f}"))
    a("")

    # 2. 초기 피처
    a("## 2. Initial Features\n")
    a(f"### Base Features ({len(BASE_FEATURES)})\n")
    a(_bullets(BASE_FEATURES))
    a("")
    a(f"### Boundary Features ({len(BOUNDARY_FEATURES)})\n")
    a(_bullets(BOUNDARY_FEATURES))
    a("")
    a(f"### Identity Features ({len(IDENTITY_FEATURES)})\n")
    a(_bullets(IDENTITY_FEATURES))
    a("")

    # 3. boundary
    a("## 3. Boundary Feature Selection\n")
    a("Base feature 고정, boundary 후보의 모든 조합을 train CV 로 비교한 결과입니다.\n")
    a(_md_table(ctx["boundary_table"]))
    a("")
    a("### Selected Boundary Features\n")
    a(_bullets(ctx["selected_boundary"]))
    a("")
    a("### Removed Boundary Features\n")
    a(_bullets(ctx["removed_boundary"]))
    a("")

    # 4. identity
    a("## 4. Identity Feature Selection\n")
    a("Base feature + 선택된 boundary feature 를 고정하고 identity 후보 조합을 비교한 결과입니다.\n")
    a(_md_table(ctx["identity_table"]))
    a("")
    a("### Selected Identity Features\n")
    a(_bullets(ctx["selected_identity"]))
    a("")
    a("### Removed Identity Features\n")
    a(_bullets(ctx["removed_identity"]))
    a("")

    # 5. 최종 피처
    a("## 5. Final Selected Features\n")
    a("### Base\n")
    a(_bullets(BASE_FEATURES))
    a("")
    a("### Boundary\n")
    a(_bullets(ctx["selected_boundary"]))
    a("")
    a("### Identity\n")
    a(_bullets(ctx["selected_identity"]))
    a("")
    a(f"**총 feature 수: {len(ctx['final_features'])}개** (후보 {len(ALL_FEATURES)}개 중)\n")

    # 6. 최종 성능
    a("## 6. Final Model Performance (test set)\n")
    a(_md_table(ctx["final_metrics_table"]))
    a("")

    # 7. 클래스별
    a("## 7. Per-Class Performance\n")
    a(_md_table(ctx["per_class"]))
    a("")
    a("```text")
    a(ctx["clf_report"].rstrip())
    a("```\n")

    # 8. confusion matrix
    a("## 8. Confusion Matrix\n")
    cm_md = ctx["cm_df"].copy()
    cm_md.insert(0, "Actual \\ Predicted", cm_md.index)
    a(_md_table(cm_md, float_fmt="{:.0f}"))
    a("")
    a("![Confusion Matrix](confusion_matrix.png)\n")

    # 9. SHAP
    a("## 9. SHAP Feature Importance\n")
    a(_md_table(ctx["shap_importance"]))
    a("")
    a("![SHAP Summary](shap_summary.png)\n")
    a("### 클래스별 mean |SHAP|\n")
    a(_md_table(ctx["shap_per_class"]))
    a("")
    for cname, fname in ctx["shap_class_plots"].items():
        a(f"![SHAP {cname}]({fname})")
    a("")

    # 10. before/after
    a("## 10. Before vs After Feature Selection\n")
    a(_md_table(ctx["compare_table"]))
    a("")

    # 11. 요약
    a("## 11. Summary\n")
    a(ctx["summary_text"])
    a("")
    return "\n".join(parts)


def build_summary_text(ctx: dict) -> str:
    lines = []
    sb = ctx["selected_boundary"]
    si = ctx["selected_identity"]
    lines.append(
        f"- Boundary 계열에서는 {', '.join(f'`{f}`' for f in sb)}가 최종 선택되었다."
        if sb
        else "- Boundary 계열에서는 성능 향상이 확인되지 않아 모든 피처가 제거되었다."
    )
    lines.append(
        f"- Identity 계열에서는 {', '.join(f'`{f}`' for f in si)}가 최종 선택되었다."
        if si
        else "- Identity 계열에서는 성능 향상이 확인되지 않아 모든 피처가 제거되었다."
    )
    lines.append(
        f"- 전체 feature 수는 {len(ALL_FEATURES)}개에서 {len(ctx['final_features'])}개로 "
        f"{'감소' if len(ctx['final_features']) < len(ALL_FEATURES) else '유지'}하였다."
    )
    all_m, sel_m = ctx["all_metrics"], ctx["final_metrics"]
    delta = sel_m["macro_f1"] - all_m["macro_f1"]
    lines.append(
        f"- test set Macro F1 은 전체 feature 모델 {all_m['macro_f1']:.4f} 에서 "
        f"선택 feature 모델 {sel_m['macro_f1']:.4f} 로 {delta:+.4f} 변화하였다."
    )
    lines.append(
        f"- 최종 모델의 Accuracy 는 {sel_m['accuracy']:.4f}, Macro AUC 는 {sel_m['macro_auc']:.4f}, "
        f"Weighted AUC 는 {sel_m['weighted_auc']:.4f} 를 기록하였다."
    )
    top = ctx["shap_importance"].iloc[0]
    lines.append(
        f"- SHAP 분석에서 가장 영향력이 높은 feature 는 `{top['feature']}` "
        f"(mean |SHAP| = {top['mean_abs_shap']:.4f}) 였다."
    )
    if delta >= -ctx["tol"]:
        lines.append(
            "- 피처 수를 줄였음에도 성능이 유지되거나 향상되어, 단순화된 feature 구성이 타당함을 확인하였다."
        )
    else:
        lines.append(
            "- 피처 축소로 test 성능이 다소 하락하였으므로, tol 값을 낮추거나 데이터를 늘려 재검토가 필요하다."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_class_weight(spec: str) -> dict:
    multipliers = {}
    for pair in spec.split(","):
        k, v = pair.split("=")
        multipliers[k.strip()] = float(v)
    return multipliers


def find_latest_csv() -> str:
    results_dir = Path(__file__).resolve().parent / "results"
    candidates = sorted(results_dir.glob("unified_features_*.csv"))
    if not candidates:
        raise SystemExit("results/unified_features_*.csv 를 찾을 수 없습니다. --csv 로 직접 지정하세요.")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(
        description="boundary/identity 피처 조합 비교 실험 (분류기 본체가 아님)"
    )
    parser.add_argument("--csv", default=None, help="입력 CSV 경로 (기본: results 의 최신 unified_features_*.csv)")
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR), help="진짜 영상 폴더")
    parser.add_argument("--test-size", type=float, default=0.2, help="최종 평가용 test 비율 (기본 0.2)")
    parser.add_argument("--n-splits", type=int, default=5, help="피처 선택 교차검증 fold 수 (기본 5)")
    parser.add_argument("--tol", type=float, default=0.005,
                        help="Macro F1 허용 오차. 이 안에 들면 더 적은 피처를 선택 (기본 0.005)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--class-weight", default=None, help="예: 'diffusion=2.0'")
    parser.add_argument("--remove-outliers", action="store_true", help="IQR 이상치 제거 (기본 미적용)")
    parser.add_argument("--out-root", default="results", help="결과 디렉터리의 상위 경로")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"select_features_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tee = Tee(sys.stdout, out_dir / "run_log.txt")
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        _run(args, csv_path, out_dir, stamp)
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"\nResults saved to:\n{out_dir.as_posix()}/")


def _run(args, csv_path: str, out_dir: Path, stamp: str):
    class_weight_multipliers = parse_class_weight(args.class_weight) if args.class_weight else {}

    print(f"[로드] {csv_path}")
    df = load_and_label(csv_path, real_dir=args.real_dir)
    if args.remove_outliers:
        df = remove_outliers_iqr(df, ALL_FEATURES)

    counts = df.groupby("group").size()
    print("\n[클래스별 샘플 수]")
    print(counts.to_string())

    y_all = df["group"].values
    df_train, df_test, y_train, y_test = train_test_split(
        df,
        y_all,
        test_size=args.test_size,
        stratify=y_all,
        random_state=args.random_state,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"\n[데이터 분리] 전체 {len(df)}개 -> train {len(df_train)} / test {len(df_test)}")
    print("train:", pd.Series(y_train).value_counts().to_dict())
    print("test :", pd.Series(y_test).value_counts().to_dict())
    if class_weight_multipliers:
        print(f"[클래스 가중치 배수] {class_weight_multipliers}")

    # --- 1) boundary 선택 -------------------------------------------------
    selected_boundary, boundary_table = select_feature_subset(
        "Boundary",
        BASE_FEATURES,
        BOUNDARY_FEATURES,
        df_train,
        y_train,
        args.tol,
        args.n_splits,
        args.random_state,
        class_weight_multipliers,
    )

    # --- 2) identity 선택 -------------------------------------------------
    selected_identity, identity_table = select_feature_subset(
        "Identity",
        BASE_FEATURES + selected_boundary,
        IDENTITY_FEATURES,
        df_train,
        y_train,
        args.tol,
        args.n_splits,
        args.random_state,
        class_weight_multipliers,
    )

    removed_boundary = [f for f in BOUNDARY_FEATURES if f not in selected_boundary]
    removed_identity = [f for f in IDENTITY_FEATURES if f not in selected_identity]

    selection_csv = pd.concat([boundary_table, identity_table], ignore_index=True)
    selection_csv.to_csv(out_dir / "feature_selection_results.csv", index=False, encoding="utf-8-sig")

    final_features = BASE_FEATURES + selected_boundary + selected_identity
    print("\nFinal selected features:")
    for f in final_features:
        print(f"- {f}")
    print(f"(총 {len(final_features)}개 / 후보 {len(ALL_FEATURES)}개)")

    # --- 3) 최종 모델 학습 + test 평가 ------------------------------------
    print("\n=== 최종 모델 학습 (train) 및 test set 평가 ===")
    final_model = fit_model(df_train, y_train, final_features, args.random_state, class_weight_multipliers)
    y_pred, proba, final_metrics = evaluate_on_test(final_model, df_test, y_test, final_features)

    metric_rows = [
        ("Accuracy", final_metrics["accuracy"]),
        ("Macro Precision", final_metrics["precision_macro"]),
        ("Macro Recall", final_metrics["recall_macro"]),
        ("Macro F1", final_metrics["macro_f1"]),
        ("Weighted Precision", final_metrics["precision_weighted"]),
        ("Weighted Recall", final_metrics["recall_weighted"]),
        ("Weighted F1", final_metrics["weighted_f1"]),
        ("Macro AUC", final_metrics["macro_auc"]),
        ("Weighted AUC", final_metrics["weighted_auc"]),
    ]
    final_metrics_table = pd.DataFrame(metric_rows, columns=["Metric", "Score"])
    print(final_metrics_table.round(4).to_string(index=False))

    clf_report = classification_report(
        y_test,
        y_pred,
        labels=DISPLAY_ORDER,
        target_names=[DISPLAY_LABEL[c] for c in DISPLAY_ORDER],
        digits=3,
        zero_division=0,
    )
    print("\n=== Classification Report (test set) ===")
    print(clf_report)

    per_class = per_class_table(y_test, y_pred)
    cm_df = save_confusion_matrix(y_test, y_pred, out_dir / "confusion_matrix.png")
    print("=== Confusion Matrix (행=실제, 열=예측) ===")
    print(cm_df.to_string())

    final_metrics_table.to_csv(out_dir / "final_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    pred_df = df_test[["video_name"]].copy()
    pred_df["true_label"] = y_test
    pred_df["predicted"] = y_pred
    for i, c in enumerate(CLASS_NAMES):
        pred_df[f"prob_{c}"] = proba[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(
        {"model": final_model, "feature_cols": final_features, "class_names": CLASS_NAMES},
        out_dir / "xgb_model_selected_features.joblib",
    )

    # --- 4) 전체 피처 모델과 비교 -----------------------------------------
    print("\n=== 전체 피처 모델 vs 선택 피처 모델 (동일 test set) ===")
    all_model = fit_model(df_train, y_train, ALL_FEATURES, args.random_state, class_weight_multipliers)
    _, _, all_metrics = evaluate_on_test(all_model, df_test, y_test, ALL_FEATURES)
    compare_table = pd.DataFrame(
        [
            {
                "Model": "All Features",
                "Feature Count": len(ALL_FEATURES),
                "Accuracy": all_metrics["accuracy"],
                "Macro F1": all_metrics["macro_f1"],
                "Macro AUC": all_metrics["macro_auc"],
            },
            {
                "Model": "Selected Features",
                "Feature Count": len(final_features),
                "Accuracy": final_metrics["accuracy"],
                "Macro F1": final_metrics["macro_f1"],
                "Macro AUC": final_metrics["macro_auc"],
            },
        ]
    )
    print(compare_table.round(4).to_string(index=False))
    compare_table.to_csv(out_dir / "before_after_comparison.csv", index=False, encoding="utf-8-sig")

    # --- 5) SHAP ----------------------------------------------------------
    shap_importance, shap_per_class, shap_class_plots = run_shap_analysis(
        final_model, df_test, final_features, out_dir
    )

    # --- 6) Markdown 보고서 ------------------------------------------------
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

    ctx = {
        "timestamp": datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
        "csv_path": csv_path,
        "tol": args.tol,
        "n_splits": args.n_splits,
        "class_weight": class_weight_multipliers or None,
        "dropped": [],
        "n_total": len(df),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "class_counts": class_counts,
        "boundary_table": boundary_table.drop(columns=["feature_group"]),
        "identity_table": identity_table.drop(columns=["feature_group"]),
        "selected_boundary": selected_boundary,
        "removed_boundary": removed_boundary,
        "selected_identity": selected_identity,
        "removed_identity": removed_identity,
        "final_features": final_features,
        "final_metrics_table": final_metrics_table,
        "final_metrics": final_metrics,
        "all_metrics": all_metrics,
        "per_class": per_class,
        "clf_report": clf_report,
        "cm_df": cm_df,
        "shap_importance": shap_importance,
        "shap_per_class": shap_per_class,
        "shap_class_plots": shap_class_plots,
        "compare_table": compare_table,
    }
    ctx["summary_text"] = build_summary_text(ctx)

    md_path = out_dir / "select_features_results.md"
    md_path.write_text(build_markdown_report(ctx), encoding="utf-8")

    print("\n=== 요약 ===")
    print(ctx["summary_text"])
    print(f"\n[저장 완료] Markdown 보고서 -> {md_path.as_posix()}")


if __name__ == "__main__":
    main()
