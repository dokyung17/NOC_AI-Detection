"""
real / diffusion / deepfake 분류 파이프라인 (XGBoost + SHAP)
================================================================

사용법:
    # 3진 분류 (real / diffusion / deepfake)
    python train_classifier.py --csv unified_features_20260716_011350.csv

    # 2진 분류 (real vs fake, diffusion+deepfake 통합)
    python train_classifier.py --csv unified_features_20260716_011350.csv --binary

필요 패키지 (로컬에서 설치):
    pip install xgboost shap scikit-learn pandas numpy joblib

이 스크립트가 하는 일:
    1. CSV 로드 + 중복 컬럼(피처가 프레임 수만큼 반복 저장된 버그) 정리
    2. 파일명 패턴으로 real / diffusion / deepfake 라벨링
       --binary 옵션을 주면 diffusion+deepfake를 'fake' 하나로 합쳐서 real vs fake 이진 분류로 진행
    3. (옵션) IQR 기반 이상치 제거 -- 기본은 꺼져 있음
    4. XGBoost 분류 (multi:softprob) 5-fold 층화 교차검증으로 성능 평가
    5. 전체 데이터로 최종 모델 학습 + 모델/클래스명 저장 (joblib)
    6. SHAP으로 전역 피처 중요도 + 개별 영상 예측 설명 출력
"""

import argparse
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# 지금까지 분석에서 확정한 7개 핵심 피처
FEATURE_COLS = [
    "absdiff_std",
    "bvp_std",
    "d3_temporal_score",
    "highfreq_score",
    "patch_corr_mean",
    "patch_signal_std_mean",
    "pseudo_snr_db",
]


# ---------------------------------------------------------------------------
# 클래스 인코딩/디코딩 (3진 / 2진 모드에 따라 class_names가 달라지므로 함수로 처리)
# ---------------------------------------------------------------------------
def build_class_maps(class_names):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def encode_labels(y_str: np.ndarray, class_to_idx: dict) -> np.ndarray:
    return np.array([class_to_idx[v] for v in y_str])


def decode_labels(y_idx: np.ndarray, idx_to_class: dict) -> np.ndarray:
    return np.array([idx_to_class[int(v)] for v in y_idx])


# ---------------------------------------------------------------------------
# 1) 데이터 로드 + 라벨링
# ---------------------------------------------------------------------------
def load_and_label(csv_path: str) -> pd.DataFrame:
    """CSV를 읽고, 중복 컬럼을 대표값 하나로 정리한 뒤 3그룹으로 라벨링한다."""
    df = pd.read_csv(csv_path)

    reduced = df[["video_name"]].copy()
    if "label" in df.columns:
        reduced["label"] = df["label"]
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            raise ValueError(f"'{feat}' 컬럼을 CSV에서 찾을 수 없습니다.")
        reduced[feat] = df[feat]

    reduced["group"] = reduced["video_name"].apply(classify_video_name)
    return reduced


def classify_video_name(name: str) -> str:
    """
    파일명 패턴으로 real / diffusion / deepfake 분류.
    (sora_/seedance_/veo_/grok_/kling_/hf_ = diffusion,
     '__' 포함 또는 idN_idM_xxxx.mp4 패턴 = deepfake(페이스스왑), 나머지 = real)
    """
    n = name.lower()
    if n.startswith("hf_") or n.startswith(("sora_", "seedance_", "veo_", "grok_", "kling_")):
        return "diffusion"
    if "__" in name or re.match(r"^id\d+_id\d+_\d+\.mp4$", name):
        return "deepfake"
    return "real"


def to_binary_group(group_series: pd.Series) -> pd.Series:
    """diffusion + deepfake -> 'fake' 로 합쳐서 real vs fake 이진 라벨로 변환."""
    return group_series.where(group_series == "real", "fake")


def classify_subgroup(name: str) -> str:
    """
    classify_video_name보다 더 세부적으로, 어떤 생성모델/기법인지까지 구분한다.
    (오분류가 특정 생성모델에 몰려있는지 진단할 때 사용)
    """
    n = name.lower()
    for prefix in ("hf_", "sora_", "seedance_", "veo_", "grok_", "kling_"):
        if n.startswith(prefix):
            return prefix.rstrip("_") if prefix != "hf_" else "hf(renamed diffusion)"
    if "__" in name:
        return "faceforensics(__)"
    if re.match(r"^id\d+_id\d+_\d+\.mp4$", name):
        return "id_id(faceswap)"
    if re.match(r"^id\d+_\d+\.mp4$", name):
        return "real(id_pattern)"
    if re.match(r"^subject\d+\.mp4$", name):
        return "real(subject_pattern)"
    if re.match(r"^\d+\.mp4$", name):
        return "real(numeric)"
    return "기타"


# ---------------------------------------------------------------------------
# 2) (옵션) IQR 이상치 제거
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3) 교차검증 평가
# ---------------------------------------------------------------------------
def evaluate_cv(X: np.ndarray, y: np.ndarray, class_names, n_splits: int = 5, random_state: int = 42,
                 class_weight_multipliers: dict = None, verbose: bool = True):
    """
    class_weight_multipliers: {클래스명: 배수} 형태로 넘기면, 기본 balanced 가중치에
    추가로 곱해서 특정 클래스를 더/덜 중요하게 학습시킬 수 있다.
    예: {'diffusion': 2.0} -> diffusion 클래스의 sample_weight를 2배로.
    verbose=False로 주면 fold별 로그/리포트를 출력하지 않고 결과만 반환한다 (그리드서치용).
    """
    from xgboost import XGBClassifier

    class_to_idx, idx_to_class = build_class_maps(class_names)
    n_classes = len(class_names)
    class_weight_multipliers = class_weight_multipliers or {}

    # XGBoost는 sklearn의 class_weight='balanced'를 직접 지원하지 않으므로
    # 클래스별 sample_weight를 직접 계산해서 넘겨준다.
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    # 여기에 사용자가 지정한 배수를 곱한다 (지정 안 하면 배수 1.0)
    weight_map = {c: w * class_weight_multipliers.get(c, 1.0) for c, w in weight_map.items()}
    sample_weight = np.array([weight_map[v] for v in y])

    y_idx = encode_labels(y, class_to_idx)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def make_model():
        params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            missing=np.nan,  # XGBoost가 결측치를 네이티브로 처리
            random_state=random_state,
            n_jobs=-1,
        )
        if n_classes == 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
        return XGBClassifier(**params)

    # 수동 fold 루프 (sample_weight를 fold별로 정확히 분리하기 위해)
    y_pred_idx = np.empty_like(y_idx)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_idx), 1):
        model = make_model()
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
    """
    target_class(예: 'diffusion')의 가중치 배수를 여러 값으로 바꿔가며 5-fold CV를 반복하고,
    각 설정에서 전체 accuracy / macro F1 / 클래스별 recall을 표로 정리해서 보여준다.
    """
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


# ---------------------------------------------------------------------------
# 진단: 오분류된 영상 찾기 + 생성모델/기법별 오류율 breakdown
# ---------------------------------------------------------------------------
def run_diagnosis(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, class_names,
                   class_weight_multipliers: dict = None,
                   n_splits: int = 5, random_state: int = 42,
                   out_csv: str = "misclassified_videos.csv"):
    """
    5-fold 교차검증(학습 때와 동일한 fold 분할)으로 모든 영상에 대한 out-of-fold 예측을 만든 뒤,
      1) 오분류된 영상 전체를 확률과 함께 CSV로 저장 (확신하며 틀린 순 정렬)
      2) 파일명 세부 패턴(생성모델/기법)별 오류율을 표로 출력
    한다.
    """
    from xgboost import XGBClassifier

    class_to_idx, idx_to_class = build_class_maps(class_names)
    n_classes = len(class_names)
    class_weight_multipliers = class_weight_multipliers or {}

    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    weight_map = {c: w * class_weight_multipliers.get(c, 1.0) for c, w in weight_map.items()}
    sample_weight = np.array([weight_map[v] for v in y])
    y_idx = encode_labels(y, class_to_idx)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def make_model():
        params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            missing=np.nan, random_state=random_state, n_jobs=-1,
        )
        if n_classes == 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
        return XGBClassifier(**params)

    y_pred_idx = np.empty_like(y_idx)
    proba_all = np.zeros((len(y_idx), n_classes))
    for train_idx, test_idx in cv.split(X, y_idx):
        model = make_model()
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

    # --- 1) 생성모델/기법(subgroup)별 오류율 ---
    print("\n=== 세부 그룹(생성모델/기법)별 정확도 ===")
    summary = (
        result.groupby("subgroup")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("accuracy")
    )
    print(summary.round(3).to_string())

    # --- 2) 오분류된 영상, '확신하며 틀린' 순으로 정렬해서 저장 ---
    wrong = result[~result["correct"]].sort_values("confidence", ascending=False)
    wrong.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 오분류 영상 {len(wrong)}개 -> {out_csv}")
    print("\n=== 가장 '확신하며' 틀린 영상 top 10 ===")
    print(wrong.head(10)[["video_name", "subgroup", "true_label", "predicted", "confidence"]]
          .to_string(index=False))

    return result



def train_final_model(X: np.ndarray, y: np.ndarray, class_names, out_path: str, random_state: int = 42,
                       class_weight_multipliers: dict = None):
    from xgboost import XGBClassifier

    class_to_idx, _ = build_class_maps(class_names)
    n_classes = len(class_names)
    class_weight_multipliers = class_weight_multipliers or {}

    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    weight_map = {c: w * class_weight_multipliers.get(c, 1.0) for c, w in weight_map.items()}
    sample_weight = np.array([weight_map[v] for v in y])
    y_idx = encode_labels(y, class_to_idx)

    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        missing=np.nan,
        random_state=random_state,
        n_jobs=-1,
    )
    if n_classes == 2:
        params.update(objective="binary:logistic", eval_metric="logloss")
    else:
        params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")

    model = XGBClassifier(**params)
    model.fit(X, y_idx, sample_weight=sample_weight)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "class_names": class_names}, out_path)
    print(f"\n[저장 완료] 최종 모델 -> {out_path}")
    return model


# ---------------------------------------------------------------------------
# 5) SHAP 설명
# ---------------------------------------------------------------------------
def explain_global(model, X: np.ndarray, feature_names):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("\n=== 전역 피처 중요도 (|SHAP| 평균) ===")
    if isinstance(shap_values, list):
        # 구버전 SHAP: 클래스별로 (n_samples, n_features) 배열의 리스트
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        sv = np.abs(shap_values)
        if sv.ndim == 3:
            # 최신 SHAP 다중분류: (n_samples, n_features, n_classes)
            # -> 샘플(0)과 클래스(2) 축으로 평균내야 피처별(1) 값이 남는다
            mean_abs = sv.mean(axis=(0, 2))
        else:
            # 이진분류: (n_samples, n_features)
            mean_abs = sv.mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    for i in order:
        print(f"  {feature_names[i]:<25s} {mean_abs[i]:.4f}")

    return explainer


def explain_single_video(model, explainer, x_row: np.ndarray, feature_names, class_names):
    """
    영상 1개(피처 벡터 1행)에 대해:
      - 클래스별 확률
      - 예측 클래스로 밀어붙인 상위 피처 기여도(SHAP)
    를 사람이 읽을 수 있는 형태로 출력한다.
    """
    x_row = x_row.reshape(1, -1)
    proba = model.predict_proba(x_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = class_names[pred_idx]

    print(f"\n예측: {pred_class} (확률 {proba[pred_idx]*100:.1f}%)")
    for i, c in enumerate(class_names):
        print(f"  - {c}: {proba[i]*100:.1f}%")

    shap_values = explainer.shap_values(x_row)
    if isinstance(shap_values, list):
        # 이진분류 구버전: 리스트 없이 바로 (1, n_features)인 경우도 있어 방어적으로 처리
        contrib = shap_values[pred_idx][0] if len(class_names) > 2 else np.array(shap_values)[0]
    elif shap_values.ndim == 3:
        contrib = shap_values[0, :, pred_idx]
    else:
        # 이진분류: (1, n_features), 양수 방향이 곧 positive class(=fake, index 1) 기여도
        raw = shap_values[0]
        contrib = raw if pred_idx == 1 else -raw

    order = np.argsort(np.abs(contrib))[::-1]
    print(f"\n'{pred_class}' 예측에 대한 피처 기여도 (영향 큰 순):")
    for i in order:
        sign = "+" if contrib[i] > 0 else "-"
        print(f"  {sign} {feature_names[i]:<25s} 값={x_row[0, i]:.4f}   기여도={contrib[i]:+.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="real/diffusion/deepfake 분류 (XGBoost + SHAP)")
    parser.add_argument("--csv", required=True, help="입력 CSV 경로 (unified_features_*.csv)")
    parser.add_argument("--binary", action="store_true",
                         help="diffusion+deepfake를 'fake'로 합쳐서 real vs fake 이진 분류로 진행")
    parser.add_argument("--hierarchical", action="store_true",
                         help="1단계 real-vs-fake -> 2단계 fake만 diffusion-vs-deepfake 로 2단계 분류 (--binary와 동시 사용 불가)")
    parser.add_argument("--weight-search", action="store_true",
                         help="클래스 가중치 배수를 그리드로 바꿔가며 성능(accuracy/macro-F1/recall) 비교")
    parser.add_argument("--weight-search-target", default="diffusion",
                         help="가중치를 조절할 대상 클래스 (기본: diffusion, 3진 분류 기준)")
    parser.add_argument("--class-weight", default=None,
                         help="최종 학습에 적용할 클래스별 가중치 배수. 'diffusion=3.0' 또는 "
                              "'diffusion=3.0,real=1.2' 처럼 콤마로 여러 개 지정 가능")
    parser.add_argument("--diagnose", action="store_true",
                         help="오분류된 영상을 찾아 CSV로 저장하고, 생성모델/기법별 오류율을 출력")
    parser.add_argument("--remove-outliers", action="store_true",
                         help="IQR 기반 이상치 제거 적용 (기본: 미적용)")
    parser.add_argument("--model-out", default=None, help="최종 모델 저장 경로")
    parser.add_argument("--explain-n", type=int, default=3,
                         help="교차검증 후 예시로 설명을 출력할 영상 개수")
    args = parser.parse_args()

    print(f"[로드] {args.csv}")
    df = load_and_label(args.csv)

    if args.weight_search:
        if args.binary or args.hierarchical:
            raise SystemExit("--weight-search는 3진 분류 기준으로만 지원돼요 (--binary, --hierarchical과 동시 사용 불가).")

        print(df.groupby("group").size())
        if args.remove_outliers:
            df = remove_outliers_iqr(df, FEATURE_COLS)

        X = df[FEATURE_COLS].values.astype(float)
        y = df["group"].values
        class_names = ["real", "diffusion", "deepfake"]

        run_class_weight_search(X, y, class_names, target_class=args.weight_search_target)
        return

    if args.hierarchical:
        if args.binary:
            raise SystemExit("--hierarchical과 --binary는 동시에 쓸 수 없습니다.")

        print(df.groupby("group").size())
        if args.remove_outliers:
            df = remove_outliers_iqr(df, FEATURE_COLS)

        X = df[FEATURE_COLS].values.astype(float)
        y = df["group"].values  # real/diffusion/deepfake 원본 3-class 라벨

        print("\n=== 5-fold 교차검증 평가 (계층적 2단계 분류) ===")
        evaluate_hierarchical_cv(X, y)

        print("\n=== 전체 데이터로 최종 (1단계+2단계) 모델 학습 ===")
        model_out = args.model_out or "xgb_model_hierarchical.joblib"
        model1, model2 = train_hierarchical_final(X, y, model_out)

        print(f"\n=== 예시 영상 {args.explain_n}개에 대한 개별 설명 ===")
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X), size=min(args.explain_n, len(X)), replace=False)
        for idx in sample_idx:
            print(f"\n--- {df.iloc[idx]['video_name']} (실제 라벨: {y[idx]}) ---")
            explain_hierarchical_single(model1, model2, X[idx], FEATURE_COLS)
        return

    if args.binary:
        df["group"] = to_binary_group(df["group"])
        class_names = ["real", "fake"]
        default_model_out = "xgb_model_binary.joblib"
    else:
        class_names = ["real", "diffusion", "deepfake"]
        default_model_out = "xgb_model_3class.joblib"

    model_out = args.model_out or default_model_out

    print(df.groupby("group").size())

    if args.remove_outliers:
        df = remove_outliers_iqr(df, FEATURE_COLS)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["group"].values

    class_weight_multipliers = {}
    if args.class_weight:
        for pair in args.class_weight.split(","):
            k, v = pair.split("=")
            class_weight_multipliers[k.strip()] = float(v)
        print(f"[클래스 가중치 배수 적용] {class_weight_multipliers}")

    print(f"\n=== 5-fold 교차검증 평가 ({'이진' if args.binary else '3진'} 분류) ===")
    evaluate_cv(X, y, class_names, class_weight_multipliers=class_weight_multipliers)

    if args.diagnose:
        run_diagnosis(df, X, y, class_names, class_weight_multipliers=class_weight_multipliers)

    print("\n=== 전체 데이터로 최종 모델 학습 ===")
    model = train_final_model(X, y, class_names, model_out, class_weight_multipliers=class_weight_multipliers)

    print("\n=== SHAP 설명 ===")
    explainer = explain_global(model, X, FEATURE_COLS)

    print(f"\n=== 예시 영상 {args.explain_n}개에 대한 개별 설명 ===")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(args.explain_n, len(X)), replace=False)
    for idx in sample_idx:
        print(f"\n--- {df.iloc[idx]['video_name']} (실제 라벨: {y[idx]}) ---")
        explain_single_video(model, explainer, X[idx], FEATURE_COLS, class_names)


# ---------------------------------------------------------------------------
# 6) 계층적(2단계) 분류: 1단계 real vs fake -> 2단계 fake만 diffusion vs deepfake
# ---------------------------------------------------------------------------
def _make_binary_xgb(random_state=42):
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        missing=np.nan,
        random_state=random_state,
        n_jobs=-1,
    )


def _fit_weighted(model, X, y_str, class_to_idx):
    classes, counts = np.unique(y_str, return_counts=True)
    weight_map = {c: len(y_str) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sw = np.array([weight_map[v] for v in y_str])
    y_idx = encode_labels(y_str, class_to_idx)
    model.fit(X, y_idx, sample_weight=sw)
    return model


def evaluate_hierarchical_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """
    outer 5-fold마다:
      1단계: train fold 전체(real/fake로 합쳐서)로 real-vs-fake 모델 학습
      2단계: train fold 중 fake만 골라서 diffusion-vs-deepfake 모델 학습
      test fold에는 1단계 -> (fake로 예측된 것만) 2단계 순서로 적용
    최종 예측(real/diffusion/deepfake)을 원래 3-class 정답과 비교해 정확도를 계산한다.
    직접 3-class 모델과 공정하게 비교하기 위해, fold 나누는 기준(StratifiedKFold)은 동일하게 유지.
    """
    bin_c2i, bin_i2c = build_class_maps(["real", "fake"])
    type_c2i, type_i2c = build_class_maps(["diffusion", "deepfake"])

    y_bin_all = to_binary_group(pd.Series(y)).values

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_final_pred = np.empty(len(y), dtype=object)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        # --- 1단계 모델: real vs fake ---
        model1 = _make_binary_xgb(random_state)
        _fit_weighted(model1, X[train_idx], y_bin_all[train_idx], bin_c2i)

        # --- 2단계 모델: train fold 중 fake(diffusion/deepfake)만 사용 ---
        fake_mask_train = y_bin_all[train_idx] == "fake"
        train_fake_idx = train_idx[fake_mask_train]
        model2 = _make_binary_xgb(random_state)
        _fit_weighted(model2, X[train_fake_idx], y[train_fake_idx], type_c2i)

        # --- test fold에 순서대로 적용 ---
        pred1_idx = model1.predict(X[test_idx])
        pred1 = decode_labels(pred1_idx, bin_i2c)

        final_pred = pred1.copy().astype(object)  # 'real' 또는 'fake'로 시작
        fake_test_mask = pred1 == "fake"
        if fake_test_mask.sum() > 0:
            test_fake_idx = test_idx[fake_test_mask]
            pred2_idx = model2.predict(X[test_fake_idx])
            pred2 = decode_labels(pred2_idx, type_i2c)
            final_pred[fake_test_mask] = pred2
        # pred1 == 'real'인 항목은 그대로 'real' 유지

        y_final_pred[test_idx] = final_pred
        acc = accuracy_score(y[test_idx], final_pred)
        print(f"  fold {fold}: accuracy = {acc:.3f}")

    class_names = ["real", "diffusion", "deepfake"]
    print("\n=== 전체 5-fold 계층적(2단계) 분류 결과 ===")
    print(f"Accuracy: {accuracy_score(y, y_final_pred):.3f}")
    print(classification_report(y, y_final_pred, target_names=class_names, labels=class_names, digits=3))
    print("Confusion matrix (행=실제, 열=예측):")
    print(pd.DataFrame(confusion_matrix(y, y_final_pred, labels=class_names),
                        index=class_names, columns=class_names))


def train_hierarchical_final(X: np.ndarray, y: np.ndarray, out_path: str, random_state: int = 42):
    """전체 데이터로 1단계/2단계 모델을 각각 학습해서 하나의 번들로 저장한다."""
    bin_c2i, _ = build_class_maps(["real", "fake"])
    type_c2i, _ = build_class_maps(["diffusion", "deepfake"])

    y_bin_all = to_binary_group(pd.Series(y)).values

    model1 = _make_binary_xgb(random_state)
    _fit_weighted(model1, X, y_bin_all, bin_c2i)

    fake_mask = y_bin_all == "fake"
    model2 = _make_binary_xgb(random_state)
    _fit_weighted(model2, X[fake_mask], y[fake_mask], type_c2i)

    joblib.dump(
        {
            "stage1_model": model1,
            "stage2_model": model2,
            "feature_cols": FEATURE_COLS,
            "stage1_classes": ["real", "fake"],
            "stage2_classes": ["diffusion", "deepfake"],
        },
        out_path,
    )
    print(f"\n[저장 완료] 계층적 최종 모델 -> {out_path}")
    return model1, model2


def explain_hierarchical_single(model1, model2, x_row: np.ndarray, feature_names):
    """
    1단계(real vs fake) 확률과 2단계(diffusion vs deepfake | fake) 확률을 결합해서
    최종 3-way 확률 [real, diffusion, deepfake]을 계산하고 출력한다.
    """
    x_row = x_row.reshape(1, -1)
    p1 = model1.predict_proba(x_row)[0]  # [P(real), P(fake)]
    p_real, p_fake = p1[0], p1[1]

    p2 = model2.predict_proba(x_row)[0]  # [P(diffusion|fake), P(deepfake|fake)]
    p_diffusion = p_fake * p2[0]
    p_deepfake = p_fake * p2[1]

    probs = {"real": p_real, "diffusion": p_diffusion, "deepfake": p_deepfake}
    pred_class = max(probs, key=probs.get)

    print(f"\n예측: {pred_class} (확률 {probs[pred_class]*100:.1f}%)")
    for c, p in probs.items():
        print(f"  - {c}: {p*100:.1f}%")

    print(f"  [1단계] real {p_real*100:.1f}% vs fake {p_fake*100:.1f}%")
    if p_fake > 0.5 or pred_class != "real":
        print(f"  [2단계, fake로 판단된 경우] diffusion {p2[0]*100:.1f}% vs deepfake {p2[1]*100:.1f}%")


if __name__ == "__main__":
    main()
