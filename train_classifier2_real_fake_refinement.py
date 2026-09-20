"""
real / diffusion / deepfake 3진 분류 파이프라인 (XGBoost + SHAP)
================================================================

사용법:
    python train_classifier2.py --csv results/unified_features_20260911_024351.csv
    python train_classifier2.py --csv ... --test-size 0.2

라벨은 CSV의 binary label 컬럼을 쓰지 않는다.
    - real: data/videos/real 폴더에 있는 파일명
    - deepfake: FaceForensics식 '__'/'~' 이름, idN_idM_xxxx.mp4, 00xx_fake.mp4
    - diffusion: 나머지

이 스크립트가 하는 일:
    1. CSV 로드 + 10개 핵심 피처만 사용
       (select_features.py 선별 결과: pseudo_snr_db / identity_sim_mean / identity_sim_min 제외)
    2. real 폴더 + 파일명 패턴으로 real / diffusion / deepfake 라벨링
    3. 층화 추출로 train / test 분리 (test 는 최종 평가에만 사용)
    4. train 내부 5-fold 교차검증으로 학습 중 성능 확인
    5. train 으로만 최종 모델 학습 후 test set 평가
    6. confusion matrix / SHAP / Markdown 보고서를 timestamp 폴더에 저장
"""

import argparse
import re
import sys
import warnings
from contextlib import contextmanager
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

warnings.filterwarnings("ignore")


@contextmanager
def redirect_stdout_to_file(log_path: str | Path):
    """모든 print 출력을 텍스트 파일로만 보낸다 (터미널에는 안 씀)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    with path.open("w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            yield path
        finally:
            sys.stdout = original


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


FEATURE_COLS = [
    "absdiff_std",
    "bvp_std",
    "d3_temporal_score",
    "highfreq_score",
    "patch_corr_mean",
    "patch_signal_std_mean",
    "boundary_score_mean",
    "boundary_score_std",
    "boundary_score_max",
    "identity_sim_std",
]

# SHAP 결과를 반영한 Real/Fake 보조 분류기 기본 피처
# (현재 실험에서 영향도가 높았던 8개 피처)
BINARY_SHAP_FEATURE_COLS = [
    "identity_sim_std",
    "bvp_std",
    "boundary_score_std",
    "highfreq_score",
    "patch_signal_std_mean",
    "boundary_score_mean",
    "d3_temporal_score",
    "boundary_score_max",
]

CLASS_NAMES = ["real", "diffusion", "deepfake"]
DISPLAY_ORDER = ["real", "deepfake", "diffusion"]
DISPLAY_LABEL = {"real": "Real", "deepfake": "Deepfake", "diffusion": "Diffusion"}
DEFAULT_REAL_DIR = Path(__file__).resolve().parent / "data" / "videos" / "real"

DEEPFAKE_ID_ID = re.compile(r"^id\d+_id\d+_\d+\.mp4$", re.IGNORECASE)
DEEPFAKE_N_FAKE = re.compile(r"^\d+_fake\.mp4$", re.IGNORECASE)


def build_class_maps(class_names):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def encode_labels(y_str: np.ndarray, class_to_idx: dict) -> np.ndarray:
    return np.array([class_to_idx[v] for v in y_str])


def decode_labels(y_idx: np.ndarray, idx_to_class: dict) -> np.ndarray:
    return np.array([idx_to_class[int(v)] for v in y_idx])


def load_real_names(real_dir: str | Path) -> set[str]:
    real_path = Path(real_dir)
    if not real_path.is_dir():
        raise SystemExit(f"real 폴더를 찾을 수 없습니다: {real_path}")
    names = {p.name.lower() for p in real_path.iterdir() if p.is_file()}
    if not names:
        raise SystemExit(f"real 폴더가 비어 있습니다: {real_path}")
    print(f"[real 폴더] {real_path} -> {len(names)}개")
    return names


def load_and_label(csv_path: str, real_dir: str | Path = DEFAULT_REAL_DIR) -> pd.DataFrame:
    """CSV를 읽고 핵심 피처만 남긴 뒤, real 폴더 + 파일명 규칙으로 3그룹 라벨을 붙인다."""
    df = pd.read_csv(csv_path)
    real_names = load_real_names(real_dir)

    reduced = df[["video_name"]].copy()
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            raise ValueError(f"'{feat}' 컬럼을 CSV에서 찾을 수 없습니다.")
        reduced[feat] = df[feat]

    reduced["group"] = reduced["video_name"].apply(lambda n: classify_video_name(n, real_names))
    return reduced


def classify_video_name(name: str, real_names: set[str] | None = None) -> str:
    """
    1) data/videos/real 에 있는 파일명 -> real
    2) 딥페이크 패턴 -> deepfake
       - FaceForensics식 '__' 또는 source~target 이름
       - idN_idM_xxxx.mp4 (예: id0_id1_0000 ~ id0_id28_0001)
       - 00xx_fake.mp4 (예: 0000_fake.mp4)
    3) 나머지 -> diffusion
    """
    fname = Path(str(name)).name
    key = fname.lower()
    if real_names is not None and key in real_names:
        return "real"
    if "__" in fname or "~" in fname or DEEPFAKE_ID_ID.match(fname) or DEEPFAKE_N_FAKE.match(fname):
        return "deepfake"
    return "diffusion"


def classify_subgroup(name: str) -> str:
    """생성모델/기법까지 나눠 오분류가 어디에 몰리는지 볼 때 사용."""
    fname = Path(str(name)).name
    n = fname.lower()
    for prefix in ("hf_", "sora_", "seedance_", "veo_", "grok_", "kling_"):
        if n.startswith(prefix):
            return prefix.rstrip("_") if prefix != "hf_" else "hf(renamed diffusion)"
    if DEEPFAKE_N_FAKE.match(fname):
        return "deepfake(N_fake)"
    if DEEPFAKE_ID_ID.match(fname):
        return "id_id(faceswap)"
    if "__" in fname or "~" in fname:
        return "faceforensics(__/~)"
    if re.match(r"^id\d+_\d+\.mp4$", n):
        return "real(id_pattern)"
    if re.match(r"^subject\d+\.mp4$", n):
        return "real(subject_pattern)"
    if re.match(r"^\d+\.mp4$", n):
        return "real(numeric)"
    return "기타"


def remove_outliers_iqr(df: pd.DataFrame, cols, k: float = 1.5) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask &= df[c].between(lo, hi) | df[c].isna()
    removed = (~mask).sum()
    print(f"[이상치 제거] {len(df)}개 -> {mask.sum()}개 ({removed}개 제거, k={k})")
    return df[mask]


def _make_multiclass_xgb(n_classes: int, random_state: int = 42):
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
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
    )


def _sample_weights(y: np.ndarray, class_weight_multipliers: dict = None) -> np.ndarray:
    class_weight_multipliers = class_weight_multipliers or {}
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    weight_map = {c: w * class_weight_multipliers.get(c, 1.0) for c, w in weight_map.items()}
    return np.array([weight_map[v] for v in y])


def compute_metrics(y_true_str: np.ndarray, y_pred_str: np.ndarray, proba: np.ndarray) -> dict:
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


def evaluate_cv(X: np.ndarray, y: np.ndarray, class_names, n_splits: int = 5, random_state: int = 42,
                 class_weight_multipliers: dict = None, verbose: bool = True):
    class_to_idx, idx_to_class = build_class_maps(class_names)
    n_classes = len(class_names)
    sample_weight = _sample_weights(y, class_weight_multipliers)
    y_idx = encode_labels(y, class_to_idx)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred_idx = np.empty_like(y_idx)
    proba_all = np.zeros((len(y_idx), n_classes))
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_idx), 1):
        model = _make_multiclass_xgb(n_classes, random_state)
        model.fit(X[train_idx], y_idx[train_idx], sample_weight=sample_weight[train_idx])
        y_pred_idx[test_idx] = model.predict(X[test_idx])
        proba_all[test_idx] = model.predict_proba(X[test_idx])
        if verbose:
            acc = accuracy_score(y_idx[test_idx], y_pred_idx[test_idx])
            print(f"  fold {fold}: accuracy = {acc:.3f}")

    y_pred = decode_labels(y_pred_idx, idx_to_class)
    metrics = compute_metrics(y, y_pred, proba_all)

    if verbose:
        print("\n=== train 내부 5-fold 교차검증 결과 ===")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1: {metrics['macro_f1']:.3f}")
        print(f"Macro AUC: {metrics['macro_auc']:.3f}")
        print(classification_report(y, y_pred, target_names=class_names, labels=class_names, digits=3))
        print("Confusion matrix (행=실제, 열=예측):")
        print(pd.DataFrame(confusion_matrix(y, y_pred, labels=class_names),
                            index=class_names, columns=class_names))

    return y, y_pred, metrics


def run_class_weight_search(X: np.ndarray, y: np.ndarray, class_names, target_class: str,
                             multipliers=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
                             n_splits: int = 5, random_state: int = 42):
    from sklearn.metrics import f1_score, recall_score

    rows = []
    for m in multipliers:
        y_true, y_pred, _ = evaluate_cv(
            X, y, class_names,
            n_splits=n_splits, random_state=random_state,
            class_weight_multipliers={target_class: m},
            verbose=False,
        )
        row = {
            "multiplier": m,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=class_names),
        }
        recalls = recall_score(y_true, y_pred, average=None, labels=class_names)
        for c, r in zip(class_names, recalls):
            row[f"recall_{c}"] = r
        rows.append(row)

    result = pd.DataFrame(rows)
    print(f"\n=== '{target_class}' 클래스 가중치 배수별 성능 비교 (나머지 클래스는 배수 1.0 고정) ===")
    print(result.round(3).to_string(index=False))

    best_recall = result.loc[result[f"recall_{target_class}"].idxmax()]
    best_f1 = result.loc[result["macro_f1"].idxmax()]
    print(f"\n[{target_class} recall 최고] multiplier={best_recall['multiplier']}, "
          f"recall_{target_class}={best_recall[f'recall_{target_class}']:.3f}, "
          f"accuracy={best_recall['accuracy']:.3f}, macro_f1={best_recall['macro_f1']:.3f}")
    print(f"[macro F1 최고] multiplier={best_f1['multiplier']}, "
          f"accuracy={best_f1['accuracy']:.3f}, macro_f1={best_f1['macro_f1']:.3f}")

    return result


def run_diagnosis(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, class_names,
                   class_weight_multipliers: dict = None,
                   n_splits: int = 5, random_state: int = 42,
                   out_csv: str = "misclassified_videos.csv"):
    class_to_idx, idx_to_class = build_class_maps(class_names)
    n_classes = len(class_names)
    sample_weight = _sample_weights(y, class_weight_multipliers)
    y_idx = encode_labels(y, class_to_idx)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred_idx = np.empty_like(y_idx)
    proba_all = np.zeros((len(y_idx), n_classes))
    for train_idx, test_idx in cv.split(X, y_idx):
        model = _make_multiclass_xgb(n_classes, random_state)
        model.fit(X[train_idx], y_idx[train_idx], sample_weight=sample_weight[train_idx])
        y_pred_idx[test_idx] = model.predict(X[test_idx])
        proba_all[test_idx] = model.predict_proba(X[test_idx])

    y_pred = decode_labels(y_pred_idx, idx_to_class)

    result = df[["video_name"]].copy()
    result["subgroup"] = result["video_name"].apply(classify_subgroup)
    result["true_label"] = y
    result["predicted"] = y_pred
    result["correct"] = result["true_label"] == result["predicted"]
    for i, c in enumerate(class_names):
        result[f"prob_{c}"] = proba_all[:, i]
    result["confidence"] = proba_all.max(axis=1)

    print("\n=== 세부 그룹(생성모델/기법)별 정확도 ===")
    summary = (
        result.groupby("subgroup")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("accuracy")
    )
    print(summary.round(3).to_string())

    wrong = result[~result["correct"]].sort_values("confidence", ascending=False)
    wrong.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 오분류 영상 {len(wrong)}개 -> {out_csv}")
    print("\n=== 가장 '확신하며' 틀린 영상 top 10 ===")
    print(wrong.head(10)[["video_name", "subgroup", "true_label", "predicted", "confidence"]]
          .to_string(index=False))

    return result


def train_final_model(X: np.ndarray, y: np.ndarray, class_names, out_path: str, random_state: int = 42,
                       class_weight_multipliers: dict = None):
    class_to_idx, _ = build_class_maps(class_names)
    n_classes = len(class_names)
    sample_weight = _sample_weights(y, class_weight_multipliers)
    y_idx = encode_labels(y, class_to_idx)

    model = _make_multiclass_xgb(n_classes, random_state)
    model.fit(X, y_idx, sample_weight=sample_weight)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "class_names": class_names}, out_path)
    print(f"\n[저장 완료] 최종 모델 -> {out_path}")
    return model


def explain_global(model, X: np.ndarray, feature_names):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("\n=== 전역 피처 중요도 (|SHAP| 평균) ===")
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        sv = np.abs(shap_values)
        if sv.ndim == 3:
            mean_abs = sv.mean(axis=(0, 2))
        else:
            mean_abs = sv.mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    for i in order:
        print(f"  {feature_names[i]:<25s} {mean_abs[i]:.4f}")

    return explainer


def explain_single_video(model, explainer, x_row: np.ndarray, feature_names, class_names):
    x_row = x_row.reshape(1, -1)
    proba = model.predict_proba(x_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = class_names[pred_idx]

    print(f"\n예측: {pred_class} (확률 {proba[pred_idx]*100:.1f}%)")
    for i, c in enumerate(class_names):
        print(f"  - {c}: {proba[i]*100:.1f}%")

    shap_values = explainer.shap_values(x_row)
    if isinstance(shap_values, list):
        contrib = shap_values[pred_idx][0]
    elif shap_values.ndim == 3:
        contrib = shap_values[0, :, pred_idx]
    else:
        contrib = shap_values[0]

    order = np.argsort(np.abs(contrib))[::-1]
    print(f"\n'{pred_class}' 예측에 대한 피처 기여도 (영향 큰 순):")
    for i in order:
        sign = "+" if contrib[i] > 0 else "-"
        print(f"  {sign} {feature_names[i]:<25s} 값={x_row[0, i]:.4f}   기여도={contrib[i]:+.4f}")


def parse_class_weight(spec: str) -> dict:
    multipliers = {}
    for pair in spec.split(","):
        k, v = pair.split("=")
        multipliers[k.strip()] = float(v)
    return multipliers


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
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black", fontsize=13,
            )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return cm_df


def _shap_array(model, X: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:
        return values[:, :, None]
    return values


def run_shap_analysis(model, X: np.ndarray, features: list[str], out_dir: Path):
    import shap

    sv = _shap_array(model, X)
    mean_abs = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1]
    importance = pd.DataFrame(
        {
            "rank": range(1, len(features) + 1),
            "feature": [features[i] for i in order],
            "mean_abs_shap": [mean_abs[i] for i in order],
        }
    )
    per_class = pd.DataFrame({"feature": features})
    for ci, cname in enumerate(CLASS_NAMES):
        per_class[f"mean_abs_shap_{cname}"] = np.abs(sv[:, :, ci]).mean(axis=0)
    per_class["mean_abs_shap_overall"] = mean_abs
    per_class = per_class.sort_values("mean_abs_shap_overall", ascending=False)

    print("\n=== SHAP Feature Importance (test set) ===")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<28s} {row['mean_abs_shap']:.4f}")
    print("\n=== 클래스별 mean |SHAP| ===")
    print(per_class.round(4).to_string(index=False))

    importance.to_csv(out_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "shap_importance_per_class.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(features) + 2))
    ax.barh([features[i] for i in order][::-1], [mean_abs[i] for i in order][::-1], color="#4c72b0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("SHAP Feature Importance (mean over classes, test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close(fig)

    class_plots = {}
    for ci, cname in enumerate(CLASS_NAMES):
        fig = plt.figure()
        shap.summary_plot(
            sv[:, :, ci], features=X, feature_names=features, show=False,
            plot_size=(7, 0.45 * len(features) + 2),
        )
        plt.title(f"SHAP summary - class: {cname}")
        fname = f"shap_summary_{cname}.png"
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        class_plots[cname] = fname
    return importance, per_class, class_plots


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
    a("# train_classifier2 Experiment Results\n")
    a(f"생성 시각: {ctx['timestamp']}\n")
    a("## 1. Experiment Setup\n")
    a("- Task: Real / Deepfake / Diffusion 3-class Classification")
    a(f"- Dataset: `{ctx['csv_path']}`")
    a("- Model: XGBoost (multi:softprob, n_estimators=300, max_depth=4, lr=0.05)")
    a("- Class Weight: balanced sample weight" + (f" + {ctx['class_weight']}" if ctx["class_weight"] else ""))
    a("- Features: 고정 10개 (select_features.py 선별 결과)")
    a(f"- Validation: train 내부 {ctx['n_splits']}-Fold Stratified CV")
    a("- Test Set: 최종 평가에만 1회 사용")
    a(f"- Total Samples: {ctx['n_total']}")
    a(f"- Train / Test: {ctx['n_train']} / {ctx['n_test']}\n")
    a("클래스별 샘플 수\n")
    a(_md_table(ctx["class_counts"], float_fmt="{:.0f}"))
    a("")
    a("## 2. Features\n")
    a(_bullets(FEATURE_COLS))
    a(f"\n**총 feature 수: {len(FEATURE_COLS)}개**\n")
    a("## 3. Train CV Performance\n")
    a(_md_table(ctx["cv_metrics_table"]))
    a("")
    a("## 4. Final Model Performance (test set)\n")
    a(_md_table(ctx["final_metrics_table"]))
    a("")
    a("## 5. Per-Class Performance\n")
    a(_md_table(ctx["per_class"]))
    a("")
    a("```text")
    a(ctx["clf_report"].rstrip())
    a("```\n")
    a("## 6. Confusion Matrix\n")
    cm_md = ctx["cm_df"].copy()
    cm_md.insert(0, "Actual \\ Predicted", cm_md.index)
    a(_md_table(cm_md, float_fmt="{:.0f}"))
    a("")
    a("![Confusion Matrix](confusion_matrix.png)\n")
    a("## 7. SHAP Feature Importance\n")
    a(_md_table(ctx["shap_importance"]))
    a("")
    a("![SHAP Summary](shap_summary.png)\n")
    a("### 클래스별 mean |SHAP|\n")
    a(_md_table(ctx["shap_per_class"]))
    a("")
    for cname, fname in ctx["shap_class_plots"].items():
        a(f"![SHAP {cname}]({fname})")
    a("")
    a("## 8. Summary\n")
    a(ctx["summary_text"])
    a("")
    return "\n".join(parts)


def build_summary_text(ctx: dict) -> str:
    m = ctx["final_metrics"]
    cv = ctx["cv_metrics"]
    top = ctx["shap_importance"].iloc[0]
    lines = [
        f"- 고정 feature {len(FEATURE_COLS)}개로 train {ctx['n_train']} / test {ctx['n_test']} 를 사용했다.",
        f"- train 내부 CV Macro F1 은 {cv['macro_f1']:.4f}, Macro AUC 는 {cv['macro_auc']:.4f} 였다.",
        f"- test Accuracy 는 {m['accuracy']:.4f}, Macro F1 은 {m['macro_f1']:.4f}, "
        f"Macro AUC 는 {m['macro_auc']:.4f} 를 기록했다.",
        f"- SHAP 에서 가장 영향력이 높은 feature 는 `{top['feature']}` "
        f"(mean |SHAP| = {top['mean_abs_shap']:.4f}) 였다.",
    ]
    return "\n".join(lines)


def metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Accuracy", metrics["accuracy"]),
            ("Macro Precision", metrics["precision_macro"]),
            ("Macro Recall", metrics["recall_macro"]),
            ("Macro F1", metrics["macro_f1"]),
            ("Weighted Precision", metrics["precision_weighted"]),
            ("Weighted Recall", metrics["recall_weighted"]),
            ("Weighted F1", metrics["weighted_f1"]),
            ("Macro AUC", metrics["macro_auc"]),
            ("Weighted AUC", metrics["weighted_auc"]),
        ],
        columns=["Metric", "Score"],
    )



def _make_binary_xgb(random_state: int = 42):
    """Real(0) vs Fake(1: deepfake+diffusion) 전용 XGBoost."""
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


def print_nan_report(df: pd.DataFrame, out_csv: Path):
    """클래스별 feature NaN 비율을 출력/저장해 shortcut 가능성을 점검한다."""
    rows = []
    for group in CLASS_NAMES:
        sub = df[df["group"] == group]
        for feat in FEATURE_COLS:
            rows.append({
                "group": group,
                "feature": feat,
                "n": len(sub),
                "nan_count": int(sub[feat].isna().sum()),
                "nan_rate": float(sub[feat].isna().mean()),
            })
    report = pd.DataFrame(rows)
    pivot = report.pivot(index="feature", columns="group", values="nan_rate")
    print("\n=== 클래스별 NaN 비율 ===")
    print((pivot * 100).round(1).astype(str).add("%").to_string())
    report.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] NaN 비율 -> {out_csv.as_posix()}")
    return report


def get_multiclass_oof_proba(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                              random_state: int = 42, class_weight_multipliers: dict = None):
    """train 내부 OOF 3-class 확률. refinement 튜닝에만 사용한다."""
    class_to_idx, _ = build_class_maps(CLASS_NAMES)
    y_idx = encode_labels(y, class_to_idx)
    sw = _sample_weights(y, class_weight_multipliers)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((len(y), len(CLASS_NAMES)), dtype=float)
    for tr, va in cv.split(X, y_idx):
        m = _make_multiclass_xgb(len(CLASS_NAMES), random_state)
        m.fit(X[tr], y_idx[tr], sample_weight=sw[tr])
        oof[va] = m.predict_proba(X[va])
    return oof


def get_binary_oof_proba(X: np.ndarray, y_binary: np.ndarray, n_splits: int = 5,
                          random_state: int = 42):
    """train 내부 OOF Real/Fake 확률."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y_binary), dtype=float)
    for tr, va in cv.split(X, y_binary):
        m = _make_binary_xgb(random_state)
        # 클래스 불균형을 fold마다 자동 보정
        n_pos = max(int((y_binary[tr] == 1).sum()), 1)
        n_neg = max(int((y_binary[tr] == 0).sum()), 1)
        sw = np.where(y_binary[tr] == 1, n_neg / n_pos, 1.0)
        m.fit(X[tr], y_binary[tr], sample_weight=sw)
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def train_binary_model(X: np.ndarray, y_binary: np.ndarray, out_path: Path,
                       feature_cols, random_state: int = 42):
    m = _make_binary_xgb(random_state)
    n_pos = max(int((y_binary == 1).sum()), 1)
    n_neg = max(int((y_binary == 0).sum()), 1)
    sw = np.where(y_binary == 1, n_neg / n_pos, 1.0)
    m.fit(X, y_binary, sample_weight=sw)
    joblib.dump({
        "model": m,
        "feature_cols": list(feature_cols),
        "classes": ["real", "fake"],
    }, out_path)
    print(f"[저장 완료] Real/Fake 보조 모델 -> {out_path.as_posix()}")
    return m


def make_refined_proba(proba3: np.ndarray, p_fake_binary: np.ndarray,
                        alpha: float, threshold: float):
    """
    3-class의 Fake 확률과 binary Fake 확률을 결합한다.
    Fake로 결정된 뒤 Deepfake/Diffusion 비율은 원 3-class 모델의 상대확률을 유지한다.
    반환: refined_pred(str), refined_proba(3-class), p_fake_final
    """
    ci = {c: i for i, c in enumerate(CLASS_NAMES)}
    p_real = proba3[:, ci["real"]]
    p_diff = proba3[:, ci["diffusion"]]
    p_deep = proba3[:, ci["deepfake"]]
    p_fake3 = p_diff + p_deep
    p_fake_final = alpha * p_fake3 + (1.0 - alpha) * p_fake_binary
    p_fake_final = np.clip(p_fake_final, 0.0, 1.0)

    denom = p_diff + p_deep
    diff_share = np.divide(p_diff, denom, out=np.full_like(p_diff, 0.5), where=denom > 0)
    deep_share = 1.0 - diff_share

    refined_proba = np.zeros_like(proba3, dtype=float)
    refined_proba[:, ci["real"]] = 1.0 - p_fake_final
    refined_proba[:, ci["diffusion"]] = p_fake_final * diff_share
    refined_proba[:, ci["deepfake"]] = p_fake_final * deep_share

    is_fake = p_fake_final >= threshold
    fake_kind = np.where(p_deep >= p_diff, "deepfake", "diffusion")
    pred = np.where(is_fake, fake_kind, "real")
    return pred.astype(object), refined_proba, p_fake_final


def tune_refinement(y_train: np.ndarray, oof_proba3: np.ndarray, oof_p_fake_binary: np.ndarray):
    """test를 보지 않고 train OOF에서 alpha와 threshold를 Macro F1 기준으로 선택."""
    from sklearn.metrics import f1_score

    rows = []
    for alpha in np.arange(0.0, 1.0001, 0.05):
        for threshold in np.arange(0.30, 0.7001, 0.01):
            pred, _, _ = make_refined_proba(
                oof_proba3, oof_p_fake_binary, float(alpha), float(threshold)
            )
            macro_f1 = f1_score(y_train, pred, labels=CLASS_NAMES, average="macro", zero_division=0)
            acc = accuracy_score(y_train, pred)
            rows.append((float(alpha), float(threshold), float(macro_f1), float(acc)))
    result = pd.DataFrame(rows, columns=["alpha_3class", "fake_threshold", "macro_f1", "accuracy"])
    # Macro F1 우선, 동률이면 accuracy 우선
    best = result.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    return result, float(best["alpha_3class"]), float(best["fake_threshold"])


def binary_metrics(y_true3: np.ndarray, p_fake: np.ndarray, threshold: float = 0.5):
    from sklearn.metrics import f1_score, balanced_accuracy_score

    y_true = (y_true3 != "real").astype(int)
    y_pred = (p_fake >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, p_fake),
    }


def correction_table(df_test: pd.DataFrame, y_true, baseline_pred, refined_pred,
                     proba3, p_fake_binary, p_fake_final):
    out = df_test[["video_name"]].copy()
    out["true_label"] = y_true
    out["baseline_pred"] = baseline_pred
    out["refined_pred"] = refined_pred
    out["baseline_correct"] = out["true_label"] == out["baseline_pred"]
    out["refined_correct"] = out["true_label"] == out["refined_pred"]
    out["changed"] = out["baseline_pred"] != out["refined_pred"]
    out["change_result"] = np.select(
        [
            (~out["baseline_correct"]) & out["refined_correct"],
            out["baseline_correct"] & (~out["refined_correct"]),
            out["changed"],
        ],
        ["fixed", "broken", "changed_still_wrong"],
        default="unchanged",
    )
    for i, c in enumerate(CLASS_NAMES):
        out[f"baseline_prob_{c}"] = proba3[:, i]
    out["binary_prob_fake"] = p_fake_binary
    out["refined_prob_fake"] = p_fake_final
    return out


def main():
    parser = argparse.ArgumentParser(description="real/diffusion/deepfake 3진 분류 (XGBoost + SHAP)")
    parser.add_argument("--csv", required=True, help="입력 CSV 경로 (unified_features_*.csv)")
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR),
                         help="진짜 영상 폴더 (여기 있는 파일명만 real로 라벨링)")
    parser.add_argument("--test-size", type=float, default=0.2, help="최종 평가용 test 비율 (기본 0.2)")
    parser.add_argument("--n-splits", type=int, default=5, help="train 내부 교차검증 fold 수 (기본 5)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--weight-search", action="store_true",
                         help="클래스 가중치 배수를 그리드로 바꿔가며 성능(accuracy/macro-F1/recall) 비교")
    parser.add_argument("--weight-search-target", default="diffusion",
                         help="가중치를 조절할 대상 클래스 (기본: diffusion)")
    parser.add_argument("--class-weight", default=None,
                         help="최종 학습에 적용할 클래스별 가중치 배수. 예: 'diffusion=3.0' 또는 "
                              "'diffusion=3.0,real=1.2'")
    parser.add_argument("--diagnose", action="store_true",
                         help="오분류된 영상을 찾아 CSV로 저장하고, 생성모델/기법별 오류율을 출력")
    parser.add_argument("--remove-outliers", action="store_true",
                         help="IQR 기반 이상치 제거 적용 (기본: 미적용)")
    parser.add_argument("--model-out", default=None, help="최종 모델 저장 경로 (기본: 결과 폴더 안)")
    parser.add_argument("--explain-n", type=int, default=3,
                         help="test 영상 중 예시로 설명을 출력할 개수")
    parser.add_argument("--out-root", default="results", help="결과 디렉터리의 상위 경로")
    parser.add_argument("--log-out", default=None, help="학습 로그 txt 경로 (기본: 결과 폴더의 run_log.txt)")
    parser.add_argument("--real-fake-refinement", action="store_true",
                         help="SHAP 기반 Real/Fake 보조 분류기 + 확률 결합 refinement 적용")
    parser.add_argument("--binary-features", choices=["shap8", "all10"], default="shap8",
                         help="Real/Fake 보조모델 피처: shap8(기본) 또는 all10")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"train_classifier2_{stamp}"
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

    # NaN shortcut 여부는 학습 전 전체 분포와 함께 반드시 기록
    print_nan_report(df, out_dir / "nan_rates_by_class.csv")

    y_all = df["group"].values

    if args.weight_search:
        run_class_weight_search(
            df[FEATURE_COLS].values.astype(float), y_all, CLASS_NAMES,
            target_class=args.weight_search_target,
        )
        return

    class_weight_multipliers = {}
    if args.class_weight:
        class_weight_multipliers = parse_class_weight(args.class_weight)
        print(f"[클래스 가중치 배수 적용] {class_weight_multipliers}")

    df_train, df_test, y_train, y_test = train_test_split(
        df, y_all, test_size=args.test_size, stratify=y_all, random_state=args.random_state,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    X_train = df_train[FEATURE_COLS].values.astype(float)
    X_test = df_test[FEATURE_COLS].values.astype(float)
    print(f"\n[데이터 분리] 전체 {len(df)}개 -> train {len(df_train)} / test {len(df_test)}")
    print("train:", pd.Series(y_train).value_counts().to_dict())
    print("test :", pd.Series(y_test).value_counts().to_dict())

    print("\n=== train 내부 5-fold 교차검증 ===")
    _, _, cv_metrics = evaluate_cv(
        X_train, y_train, CLASS_NAMES,
        n_splits=args.n_splits, random_state=args.random_state,
        class_weight_multipliers=class_weight_multipliers,
    )

    if args.diagnose:
        run_diagnosis(
            df_train, X_train, y_train, CLASS_NAMES,
            class_weight_multipliers=class_weight_multipliers,
            n_splits=args.n_splits, random_state=args.random_state,
            out_csv=str(out_dir / "misclassified_videos.csv"),
        )

    print("\n=== train 데이터로 최종 3-class 모델 학습 ===")
    model_out = args.model_out or str(out_dir / "xgb_model_3class.joblib")
    model = train_final_model(
        X_train, y_train, CLASS_NAMES, model_out,
        random_state=args.random_state,
        class_weight_multipliers=class_weight_multipliers,
    )

    _, idx_to_class = build_class_maps(CLASS_NAMES)
    proba = model.predict_proba(X_test)
    y_pred = decode_labels(model.predict(X_test), idx_to_class)
    final_metrics = compute_metrics(y_test, y_pred, proba)
    final_metrics_table = metrics_table(final_metrics)
    cv_metrics_table = metrics_table(cv_metrics)

    print("\n=== Baseline 3-class test 평가 ===")
    print(final_metrics_table.round(4).to_string(index=False))
    clf_report = classification_report(
        y_test, y_pred, labels=DISPLAY_ORDER,
        target_names=[DISPLAY_LABEL[c] for c in DISPLAY_ORDER],
        digits=3, zero_division=0,
    )
    print("\n=== Baseline Classification Report ===")
    print(clf_report)

    per_class = per_class_table(y_test, y_pred)
    cm_df = save_confusion_matrix(y_test, y_pred, out_dir / "confusion_matrix.png")
    print("=== Baseline Confusion Matrix (행=실제, 열=예측) ===")
    print(cm_df.to_string())

    final_metrics_table.to_csv(out_dir / "final_metrics.csv", index=False, encoding="utf-8-sig")
    cv_metrics_table.to_csv(out_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    pred_df = df_test[["video_name"]].copy()
    pred_df["true_label"] = y_test
    pred_df["predicted"] = y_pred
    for i, c in enumerate(CLASS_NAMES):
        pred_df[f"prob_{c}"] = proba[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    refined_metrics = None
    if args.real_fake_refinement:
        binary_cols = BINARY_SHAP_FEATURE_COLS if args.binary_features == "shap8" else FEATURE_COLS
        print("\n=== Real/Fake refinement ===")
        print(f"[Binary features: {args.binary_features}] {binary_cols}")
        Xb_train = df_train[binary_cols].values.astype(float)
        Xb_test = df_test[binary_cols].values.astype(float)
        yb_train = (y_train != "real").astype(int)

        print("[1/4] train OOF 3-class / Real-Fake 확률 생성")
        oof_proba3 = get_multiclass_oof_proba(
            X_train, y_train, n_splits=args.n_splits, random_state=args.random_state,
            class_weight_multipliers=class_weight_multipliers,
        )
        oof_p_fake_binary = get_binary_oof_proba(
            Xb_train, yb_train, n_splits=args.n_splits, random_state=args.random_state,
        )
        oof_bin = binary_metrics(y_train, oof_p_fake_binary, threshold=0.5)
        print("\n=== train OOF Real/Fake 보조모델 ===")
        print(f"Accuracy          : {oof_bin['accuracy']:.4f}")
        print(f"F1                : {oof_bin['f1']:.4f}")
        print(f"Balanced Accuracy : {oof_bin['balanced_accuracy']:.4f}")
        print(f"AUC               : {oof_bin['auc']:.4f}")

        print("[2/4] train OOF에서 결합 alpha / Fake threshold 자동 탐색")
        search_df, best_alpha, best_threshold = tune_refinement(
            y_train, oof_proba3, oof_p_fake_binary
        )
        search_df.to_csv(out_dir / "refinement_search.csv", index=False, encoding="utf-8-sig")
        best_row = search_df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        print(f"[선택] alpha_3class={best_alpha:.2f}, alpha_binary={1-best_alpha:.2f}, "
              f"fake_threshold={best_threshold:.2f}")
        print(f"[train OOF] refined Macro F1={best_row['macro_f1']:.4f}, "
              f"Accuracy={best_row['accuracy']:.4f}")

        print("[3/4] train 전체로 Real/Fake 최종 보조모델 학습")
        binary_model = train_binary_model(
            Xb_train, yb_train, out_dir / "xgb_model_real_fake.joblib",
            binary_cols, random_state=args.random_state,
        )

        print("[4/4] 고정된 alpha/threshold로 test refinement 평가")
        p_fake_binary_test = binary_model.predict_proba(Xb_test)[:, 1]
        test_bin = binary_metrics(y_test, p_fake_binary_test, threshold=0.5)
        print("\n=== Binary Real/Fake test (참고용 threshold=0.50) ===")
        print(f"Accuracy          : {test_bin['accuracy']:.4f}")
        print(f"F1                : {test_bin['f1']:.4f}")
        print(f"Balanced Accuracy : {test_bin['balanced_accuracy']:.4f}")
        print(f"AUC               : {test_bin['auc']:.4f}")

        refined_pred, refined_proba, p_fake_final = make_refined_proba(
            proba, p_fake_binary_test, best_alpha, best_threshold
        )
        refined_metrics = compute_metrics(y_test, refined_pred, refined_proba)
        refined_metrics_table = metrics_table(refined_metrics)
        print("\n=== Refined 3-class test 평가 ===")
        print(refined_metrics_table.round(4).to_string(index=False))
        refined_report = classification_report(
            y_test, refined_pred, labels=DISPLAY_ORDER,
            target_names=[DISPLAY_LABEL[c] for c in DISPLAY_ORDER],
            digits=3, zero_division=0,
        )
        print("\n=== Refined Classification Report ===")
        print(refined_report)
        refined_cm = save_confusion_matrix(
            y_test, refined_pred, out_dir / "confusion_matrix_refined.png"
        )
        print("=== Refined Confusion Matrix ===")
        print(refined_cm.to_string())

        refined_metrics_table.to_csv(out_dir / "refined_metrics.csv", index=False, encoding="utf-8-sig")
        refined_cm.to_csv(out_dir / "confusion_matrix_refined.csv", encoding="utf-8-sig")
        pd.DataFrame([test_bin]).to_csv(out_dir / "binary_real_fake_metrics.csv", index=False, encoding="utf-8-sig")

        corrections = correction_table(
            df_test, y_test, y_pred, refined_pred, proba, p_fake_binary_test, p_fake_final
        )
        corrections.to_csv(out_dir / "refinement_predictions.csv", index=False, encoding="utf-8-sig")
        changed = corrections[corrections["changed"]]
        changed.to_csv(out_dir / "refinement_changed_samples.csv", index=False, encoding="utf-8-sig")
        fixed = int((corrections["change_result"] == "fixed").sum())
        broken = int((corrections["change_result"] == "broken").sum())
        print("\n=== Refinement 변화 요약 ===")
        print(f"변경된 예측: {len(changed)}개")
        print(f"오분류 -> 정답 교정: {fixed}개")
        print(f"정답 -> 오분류 악화: {broken}개")
        print(f"순 개선: {fixed - broken:+d}개")
        print(f"Accuracy: {final_metrics['accuracy']:.4f} -> {refined_metrics['accuracy']:.4f} "
              f"({refined_metrics['accuracy']-final_metrics['accuracy']:+.4f})")
        print(f"Macro F1: {final_metrics['macro_f1']:.4f} -> {refined_metrics['macro_f1']:.4f} "
              f"({refined_metrics['macro_f1']-final_metrics['macro_f1']:+.4f})")
        print(f"Macro AUC: {final_metrics['macro_auc']:.4f} -> {refined_metrics['macro_auc']:.4f} "
              f"({refined_metrics['macro_auc']-final_metrics['macro_auc']:+.4f})")

        config_df = pd.DataFrame([{
            "binary_feature_mode": args.binary_features,
            "binary_features": ",".join(binary_cols),
            "alpha_3class": best_alpha,
            "alpha_binary": 1.0 - best_alpha,
            "fake_threshold": best_threshold,
            "train_oof_binary_auc": oof_bin["auc"],
            "train_oof_refined_macro_f1": best_row["macro_f1"],
        }])
        config_df.to_csv(out_dir / "refinement_config.csv", index=False, encoding="utf-8-sig")

    # 기존 SHAP 분석은 baseline 3-class 모델에 대해 유지
    shap_importance, shap_per_class, shap_class_plots = run_shap_analysis(
        model, X_test, FEATURE_COLS, out_dir
    )
    explainer = explain_global(model, X_test, FEATURE_COLS)

    print(f"\n=== test 예시 영상 {args.explain_n}개에 대한 개별 설명 ===")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(args.explain_n, len(X_test)), replace=False)
    for idx in sample_idx:
        print(f"\n--- {df_test.iloc[idx]['video_name']} (실제 라벨: {y_test[idx]}) ---")
        explain_single_video(model, explainer, X_test[idx], FEATURE_COLS, CLASS_NAMES)

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
        "csv_path": args.csv,
        "n_splits": args.n_splits,
        "class_weight": class_weight_multipliers or None,
        "n_total": len(df),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "class_counts": class_counts,
        "cv_metrics": cv_metrics,
        "cv_metrics_table": cv_metrics_table,
        "final_metrics": final_metrics,
        "final_metrics_table": final_metrics_table,
        "per_class": per_class,
        "clf_report": clf_report,
        "cm_df": cm_df,
        "shap_importance": shap_importance,
        "shap_per_class": shap_per_class,
        "shap_class_plots": shap_class_plots,
    }
    ctx["summary_text"] = build_summary_text(ctx)
    md_path = out_dir / "train_classifier2_results.md"
    md_path.write_text(build_markdown_report(ctx), encoding="utf-8")

    print("\n=== 요약 ===")
    print(ctx["summary_text"])
    if refined_metrics is not None:
        print(f"- Refinement test Accuracy: {final_metrics['accuracy']:.4f} -> {refined_metrics['accuracy']:.4f}")
        print(f"- Refinement test Macro F1: {final_metrics['macro_f1']:.4f} -> {refined_metrics['macro_f1']:.4f}")
    print(f"\n[저장 완료] Markdown 보고서 -> {md_path.as_posix()}")


if __name__ == "__main__":
    main()
