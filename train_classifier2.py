"""
real / diffusion / deepfake 3진 분류 파이프라인 (XGBoost + SHAP)
================================================================

사용법:
    python train_classifier2.py --csv unified_features_20260811_202802.csv
    python train_classifier2.py --csv unified_features_20260811_202802.csv --log-out results/train2_log.txt

실행 로그는 터미널이 아니라 --log-out 텍스트 파일로만 저장된다.

라벨은 CSV의 binary label 컬럼을 쓰지 않는다.
    - real: data/videos/real 폴더에 있는 파일명
    - deepfake: FaceForensics식 '__'/'~' 이름, idN_idM_xxxx.mp4, 00xx_fake.mp4
    - diffusion: 나머지

이 스크립트가 하는 일:
    1. CSV 로드 + 7개 핵심 피처만 사용
    2. real 폴더 + 파일명 패턴으로 real / diffusion / deepfake 라벨링
    3. (옵션) IQR 기반 이상치 제거 -- 기본은 꺼져 있음
    4. XGBoost 분류 (multi:softprob) 5-fold 층화 교차검증으로 성능 평가
    5. 전체 데이터로 최종 모델 학습 + 모델/클래스명 저장 (joblib)
    6. SHAP으로 전역 피처 중요도 + 개별 영상 예측 설명 출력
"""

import argparse
import re
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

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

FEATURE_COLS = [
    "absdiff_std",
    "bvp_std",
    "d3_temporal_score",
    "highfreq_score",
    "patch_corr_mean",
    "patch_signal_std_mean",
    "pseudo_snr_db",
]

CLASS_NAMES = ["real", "diffusion", "deepfake"]
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
    """CSV를 읽고 7개 피처만 남긴 뒤, real 폴더 + 파일명 규칙으로 3그룹 라벨을 붙인다."""
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


def evaluate_cv(X: np.ndarray, y: np.ndarray, class_names, n_splits: int = 5, random_state: int = 42,
                 class_weight_multipliers: dict = None, verbose: bool = True):
    class_to_idx, idx_to_class = build_class_maps(class_names)
    n_classes = len(class_names)
    sample_weight = _sample_weights(y, class_weight_multipliers)
    y_idx = encode_labels(y, class_to_idx)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred_idx = np.empty_like(y_idx)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_idx), 1):
        model = _make_multiclass_xgb(n_classes, random_state)
        model.fit(X[train_idx], y_idx[train_idx], sample_weight=sample_weight[train_idx])
        y_pred_idx[test_idx] = model.predict(X[test_idx])
        if verbose:
            acc = accuracy_score(y_idx[test_idx], y_pred_idx[test_idx])
            print(f"  fold {fold}: accuracy = {acc:.3f}")

    y_pred = decode_labels(y_pred_idx, idx_to_class)

    if verbose:
        print("\n=== 전체 5-fold 교차검증 결과 ===")
        print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
        print(classification_report(y, y_pred, target_names=class_names, labels=class_names, digits=3))
        print("Confusion matrix (행=실제, 열=예측):")
        print(pd.DataFrame(confusion_matrix(y, y_pred, labels=class_names),
                            index=class_names, columns=class_names))

    return y, y_pred


def run_class_weight_search(X: np.ndarray, y: np.ndarray, class_names, target_class: str,
                             multipliers=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
                             n_splits: int = 5, random_state: int = 42):
    from sklearn.metrics import f1_score, recall_score

    rows = []
    for m in multipliers:
        y_true, y_pred = evaluate_cv(
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


def main():
    parser = argparse.ArgumentParser(description="real/diffusion/deepfake 3진 분류 (XGBoost + SHAP)")
    parser.add_argument("--csv", required=True, help="입력 CSV 경로 (unified_features_*.csv)")
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR),
                         help="진짜 영상 폴더 (여기 있는 파일명만 real로 라벨링)")
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
    parser.add_argument("--model-out", default="xgb_model_3class_0824.joblib", help="최종 모델 저장 경로")
    parser.add_argument("--explain-n", type=int, default=3,
                         help="교차검증 후 예시로 설명을 출력할 영상 개수")
    parser.add_argument(
        "--log-out",
        default=None,
        help="학습 로그 txt 경로 (기본: train_classifier2_YYYYMMDD_HHMMSS.txt)",
    )
    args = parser.parse_args()

    log_path = args.log_out or f"train_classifier2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with redirect_stdout_to_file(log_path):
        _run(args)

    # 터미널에는 저장 위치만 한 줄
    print(f"[로그 저장] {Path(log_path).resolve()}", file=sys.__stdout__)


def _run(args):
    print(f"[로드] {args.csv}")
    df = load_and_label(args.csv, real_dir=args.real_dir)
    print(df.groupby("group").size())

    if args.remove_outliers:
        df = remove_outliers_iqr(df, FEATURE_COLS)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["group"].values

    if args.weight_search:
        run_class_weight_search(X, y, CLASS_NAMES, target_class=args.weight_search_target)
        return

    class_weight_multipliers = {}
    if args.class_weight:
        class_weight_multipliers = parse_class_weight(args.class_weight)
        print(f"[클래스 가중치 배수 적용] {class_weight_multipliers}")

    print("\n=== 5-fold 교차검증 평가 (3진 분류) ===")
    evaluate_cv(X, y, CLASS_NAMES, class_weight_multipliers=class_weight_multipliers)

    if args.diagnose:
        run_diagnosis(df, X, y, CLASS_NAMES, class_weight_multipliers=class_weight_multipliers)

    print("\n=== 전체 데이터로 최종 모델 학습 ===")
    model = train_final_model(X, y, CLASS_NAMES, args.model_out,
                              class_weight_multipliers=class_weight_multipliers)

    print("\n=== SHAP 설명 ===")
    explainer = explain_global(model, X, FEATURE_COLS)

    print(f"\n=== 예시 영상 {args.explain_n}개에 대한 개별 설명 ===")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(args.explain_n, len(X)), replace=False)
    for idx in sample_idx:
        print(f"\n--- {df.iloc[idx]['video_name']} (실제 라벨: {y[idx]}) ---")
        explain_single_video(model, explainer, X[idx], FEATURE_COLS, CLASS_NAMES)


if __name__ == "__main__":
    main()
