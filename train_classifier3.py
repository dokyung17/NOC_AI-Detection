"""
이중 이진 분류기: real-vs-diffusion + real-vs-deepfake
======================================================

사용법:
    python train_classifier3.py --csv results/Green_unified_features_20260812_194745.csv
    python train_classifier3.py --csv results/Green_unified_features_20260812_194745.csv --tau-diff 0.5 --tau-df 0.5

실행 시 항상 τ 그리드 서치(0.30~0.80, step 0.05)를 돌리고
추천 임계값을 자동 적용하며, 비교표는 threshold_comparison.txt 에 저장한다.
(--tau-diff / --tau-df 를 주면 그 값을 수동 사용)

실행 로그는 터미널이 아니라 --log-out 텍스트 파일로만 저장된다.

아이디어:
    A) real + diffusion 만으로 real vs diffusion 학습
    B) real + deepfake 만으로 real vs deepfake 학습
    새 영상에는 둘 다 적용해서
      - 디퓨전-리얼 분류기: P(diffusion)
      - 딥페-리얼 분류기:   P(deepfake)
    를 각각 출력한다.

선택적으로 (--combine / 기본 켜짐) 두 점수를 임계값으로 합쳐
    real / diffusion / deepfake / ambiguous
    3진(+애매) 라벨을 만든다.

최종 A·B 모델에 대해 SHAP 전역 중요도와 예시 영상별 기여도를 로그에 남긴다.
"""

from __future__ import annotations

import argparse
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
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from train_classifier2 import (
    CLASS_NAMES,
    DEFAULT_REAL_DIR,
    FEATURE_COLS,
    load_and_label,
    remove_outliers_iqr,
)

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

PAIR_DIFF = ("real", "diffusion")  # positive = diffusion (index 1)
PAIR_DF = ("real", "deepfake")     # positive = deepfake  (index 1)


def _make_binary_xgb(random_state: int = 42):
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


def _balanced_sample_weight(y_idx: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y_idx, return_counts=True)
    weight_map = {c: len(y_idx) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y_idx])


def _encode_binary(y_str: np.ndarray, positive: str) -> np.ndarray:
    """positive 클래스 = 1, real = 0."""
    return (y_str == positive).astype(int)


def subset_mask(y: np.ndarray, keep: tuple[str, str]) -> np.ndarray:
    return np.isin(y, list(keep))


def evaluate_binary_cv(
    X: np.ndarray,
    y: np.ndarray,
    positive: str,
    pair_name: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    """해당 쌍(real vs positive)만으로 5-fold 이진 평가."""
    mask = subset_mask(y, ("real", positive))
    X_sub, y_sub = X[mask], y[mask]
    y_idx = _encode_binary(y_sub, positive)
    sw = _balanced_sample_weight(y_idx)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.empty_like(y_idx)
    proba_pos = np.empty(len(y_idx), dtype=float)

    print(f"\n=== [{pair_name}] 5-fold (home: real vs {positive}, n={len(y_sub)}) ===")
    for fold, (tr, te) in enumerate(cv.split(X_sub, y_idx), 1):
        model = _make_binary_xgb(random_state)
        model.fit(X_sub[tr], y_idx[tr], sample_weight=sw[tr])
        y_pred[te] = model.predict(X_sub[te])
        proba_pos[te] = model.predict_proba(X_sub[te])[:, 1]
        acc = accuracy_score(y_idx[te], y_pred[te])
        print(f"  fold {fold}: accuracy = {acc:.3f}")

    labels_str = ["real", positive]
    y_true_str = np.where(y_idx == 1, positive, "real")
    y_pred_str = np.where(y_pred == 1, positive, "real")
    print(f"Accuracy: {accuracy_score(y_true_str, y_pred_str):.3f}")
    print(classification_report(y_true_str, y_pred_str, target_names=labels_str, labels=labels_str, digits=3))
    print("Confusion matrix (행=실제, 열=예측):")
    print(pd.DataFrame(
        confusion_matrix(y_true_str, y_pred_str, labels=labels_str),
        index=labels_str, columns=labels_str,
    ))
    return proba_pos, mask


def oof_positive_proba(
    X: np.ndarray,
    y: np.ndarray,
    positive: str,
    train_classes: tuple[str, str],
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    train_classes만으로 학습하되, 전체 X에 대한 out-of-fold P(positive)를 만든다.
    (다른 클래스 행은 '그 fold의 학습 모델'로만 점수 매김 — 공정한 운영 시뮬레이션)
    """
    y_bin_all = np.where(y == "real", "real", np.where(y == positive, positive, "other"))
    # fold는 3-class 기준으로 나눠서 모든 영상이 한 번씩 test에 들어가게 함
    y_for_split = y  # real/diffusion/deepfake
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = np.full(len(y), np.nan)

    for tr, te in cv.split(X, y_for_split):
        train_mask = subset_mask(y[tr], train_classes)
        tr_idx = tr[train_mask]
        y_tr = _encode_binary(y[tr_idx], positive)
        sw = _balanced_sample_weight(y_tr)
        model = _make_binary_xgb(random_state)
        model.fit(X[tr_idx], y_tr, sample_weight=sw)
        proba[te] = model.predict_proba(X[te])[:, 1]

    assert np.isfinite(proba).all()
    return proba


def combine_scores(
    p_diff: np.ndarray,
    p_df: np.ndarray,
    tau_diff: float = 0.5,
    tau_df: float = 0.5,
    allow_ambiguous: bool = True,
) -> np.ndarray:
    """
    두 점수로 라벨 합치기.
      A만 높음 -> diffusion
      B만 높음 -> deepfake
      둘 다 낮음 -> real
      둘 다 높음 -> ambiguous (allow_ambiguous=False면 더 높은 쪽)
    """
    a = p_diff > tau_diff
    b = p_df > tau_df
    out = np.empty(len(p_diff), dtype=object)
    both = a & b
    neither = ~a & ~b
    only_a = a & ~b
    only_b = b & ~a

    out[only_a] = "diffusion"
    out[only_b] = "deepfake"
    out[neither] = "real"
    if allow_ambiguous:
        out[both] = "ambiguous"
    else:
        pick_diff = both & (p_diff >= p_df)
        out[pick_diff] = "diffusion"
        out[both & ~pick_diff] = "deepfake"
    return out


def evaluate_combined(
    y_true: np.ndarray,
    p_diff: np.ndarray,
    p_df: np.ndarray,
    tau_diff: float,
    tau_df: float,
    allow_ambiguous: bool,
):
    pred = combine_scores(p_diff, p_df, tau_diff, tau_df, allow_ambiguous=allow_ambiguous)
    n_amb = int((pred == "ambiguous").sum()) if allow_ambiguous else 0

    # ambiguous는 정확도에서 제외(또는 오답 처리) — 여기선 제외하고 coverage도 같이 보고
    if allow_ambiguous:
        decided = pred != "ambiguous"
        y_d, p_d = y_true[decided], pred[decided]
        coverage = decided.mean()
        print(f"\n=== 합친 3진 평가 (τ_diff={tau_diff}, τ_df={tau_df}, ambiguous 허용) ===")
        print(f"coverage (ambiguous 제외): {coverage:.3f}  | ambiguous: {n_amb}/{len(y_true)}")
        if decided.sum() == 0:
            print("결정된 샘플이 없습니다.")
            return pred
        print(f"Accuracy (decided only): {accuracy_score(y_d, p_d):.3f}")
        print(classification_report(y_d, p_d, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=3, zero_division=0))
        print("Confusion matrix (decided, 행=실제, 열=예측):")
        print(pd.DataFrame(confusion_matrix(y_d, p_d, labels=CLASS_NAMES),
                            index=CLASS_NAMES, columns=CLASS_NAMES))
    else:
        print(f"\n=== 합친 3진 평가 (τ_diff={tau_diff}, τ_df={tau_df}, 동시고득점→더 높은 쪽) ===")
        print(f"Accuracy: {accuracy_score(y_true, pred):.3f}")
        print(classification_report(y_true, pred, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=3, zero_division=0))
        print("Confusion matrix (행=실제, 열=예측):")
        print(pd.DataFrame(confusion_matrix(y_true, pred, labels=CLASS_NAMES),
                            index=CLASS_NAMES, columns=CLASS_NAMES))
        print(f"macro F1: {f1_score(y_true, pred, average='macro', labels=CLASS_NAMES):.3f}")
    return pred


def threshold_search(
    y_true: np.ndarray,
    p_diff: np.ndarray,
    p_df: np.ndarray,
    grid=None,
    out_txt: str | None = "threshold_comparison.txt",
):
    """
    ambiguous 없이(동시고득점→높은 쪽) τ 그리드 서치.
    macro_f1 → accuracy 순으로 최고 조합을 고른다.
    """
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.30, 0.81, 0.05)]

    rows = []
    for td in grid:
        for tf in grid:
            pred = combine_scores(p_diff, p_df, td, tf, allow_ambiguous=False)
            row = {
                "tau_diff": td,
                "tau_df": tf,
                "accuracy": accuracy_score(y_true, pred),
                "macro_f1": f1_score(y_true, pred, average="macro", labels=CLASS_NAMES),
            }
            recalls = recall_score(y_true, pred, average=None, labels=CLASS_NAMES, zero_division=0)
            for c, r in zip(CLASS_NAMES, recalls):
                row[f"recall_{c}"] = r
            # balanced recall: 세 클래스 recall의 최솟값 (한쪽으로만 치우친 해 완화)
            row["min_recall"] = float(np.min(recalls))
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy", "min_recall"], ascending=False
    ).reset_index(drop=True)
    best = result.iloc[0]
    best_tau_diff = float(best["tau_diff"])
    best_tau_df = float(best["tau_df"])

    # 0.5 기준선도 같이 적어 비교하기 쉽게
    baseline = result[(result["tau_diff"] == 0.5) & (result["tau_df"] == 0.5)]
    baseline_line = ""
    if len(baseline):
        b0 = baseline.iloc[0]
        baseline_line = (
            f"기준(0.50/0.50): accuracy={b0['accuracy']:.3f}, "
            f"macro_f1={b0['macro_f1']:.3f}, min_recall={b0['min_recall']:.3f}\n"
        )

    header = (
        "임계값 비교 (ambiguous 없음, 둘 다 높으면 더 높은 쪽 선택)\n"
        f"선정 기준: macro_f1 최대 → accuracy → min_recall\n"
        f"그리드: {grid[0]:.2f} ~ {grid[-1]:.2f} step {grid[1]-grid[0]:.2f} "
        f"({len(grid)}x{len(grid)} = {len(result)} 조합)\n"
        f"{baseline_line}"
        f"[추천] tau_diff={best_tau_diff:.2f}, tau_df={best_tau_df:.2f}  "
        f"accuracy={best['accuracy']:.3f}, macro_f1={best['macro_f1']:.3f}, "
        f"min_recall={best['min_recall']:.3f}\n"
        f"  recall_real={best['recall_real']:.3f}, "
        f"recall_diffusion={best['recall_diffusion']:.3f}, "
        f"recall_deepfake={best['recall_deepfake']:.3f}\n"
        f"{'-' * 88}\n"
        f"{'rank':>4}  {'tau_d':>5}  {'tau_df':>6}  {'acc':>6}  {'macroF1':>7}  "
        f"{'minR':>5}  {'R_real':>6}  {'R_diff':>6}  {'R_df':>6}\n"
        f"{'-' * 88}\n"
    )
    body_lines = []
    show = result.round(3)
    for i, row in show.iterrows():
        body_lines.append(
            f"{i+1:4d}  {row['tau_diff']:5.2f}  {row['tau_df']:6.2f}  "
            f"{row['accuracy']:6.3f}  {row['macro_f1']:7.3f}  {row['min_recall']:5.3f}  "
            f"{row['recall_real']:6.3f}  {row['recall_diffusion']:6.3f}  "
            f"{row['recall_deepfake']:6.3f}"
        )
    text = header + "\n".join(body_lines) + "\n"

    print("\n=== 임계값 그리드 서치 ===")
    print(f"[추천] tau_diff={best_tau_diff:.2f}, tau_df={best_tau_df:.2f} "
          f"(accuracy={best['accuracy']:.3f}, macro_f1={best['macro_f1']:.3f}, "
          f"min_recall={best['min_recall']:.3f})")
    print(show.head(10).to_string(index=False))

    if out_txt:
        Path(out_txt).write_text(text, encoding="utf-8")
        print(f"[저장] 임계값 비교표 -> {out_txt}")

    return result, best_tau_diff, best_tau_df


def train_final_pair(X: np.ndarray, y: np.ndarray, positive: str, random_state: int = 42):
    mask = subset_mask(y, ("real", positive))
    X_sub, y_sub = X[mask], y[mask]
    y_idx = _encode_binary(y_sub, positive)
    sw = _balanced_sample_weight(y_idx)
    model = _make_binary_xgb(random_state)
    model.fit(X_sub, y_idx, sample_weight=sw)
    return model


def explain_global_binary(model, X: np.ndarray, feature_names, title: str):
    """이진 모델 전역 |SHAP| 평균 (양수 방향 = positive class)."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(f"\n=== {title} 전역 피처 중요도 (|SHAP| 평균) ===")
    if isinstance(shap_values, list):
        # 구버전: [neg, pos] 리스트면 positive(class 1) 사용
        sv = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        mean_abs = sv.mean(axis=0)
    else:
        sv = np.abs(shap_values)
        if sv.ndim == 3:
            mean_abs = sv[:, :, 1].mean(axis=0) if sv.shape[-1] > 1 else sv.mean(axis=(0, 2))
        else:
            mean_abs = sv.mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    for i in order:
        print(f"  {feature_names[i]:<25s} {mean_abs[i]:.4f}")
    return explainer


def _binary_shap_contrib(explainer, x_row: np.ndarray, toward_positive: bool = True) -> np.ndarray:
    """이진 SHAP에서 positive class 방향 기여도 벡터를 반환."""
    shap_values = explainer.shap_values(x_row.reshape(1, -1))
    if isinstance(shap_values, list):
        raw = np.array(shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0])
    elif getattr(shap_values, "ndim", 1) == 3:
        raw = shap_values[0, :, 1] if shap_values.shape[-1] > 1 else shap_values[0, :, 0]
    else:
        raw = np.asarray(shap_values[0])
    return raw if toward_positive else -raw


def explain_single_binary(
    model,
    explainer,
    x_row: np.ndarray,
    feature_names,
    positive_name: str,
    tag: str,
):
    """한 영상에 대해 이진 모델 확률 + positive 방향 SHAP 기여도."""
    x = x_row.reshape(1, -1)
    p_pos = float(model.predict_proba(x)[0, 1])
    pred = positive_name if p_pos >= 0.5 else "real"
    print(f"  [{tag}] 예측={pred}  |  {positive_name} {p_pos*100:.1f}%  /  real {(1-p_pos)*100:.1f}%")

    contrib = _binary_shap_contrib(explainer, x, toward_positive=True)
    order = np.argsort(np.abs(contrib))[::-1]
    print(f"  [{tag}] '{positive_name}' 쪽으로의는 피처 기여도:")
    for i in order:
        sign = "+" if contrib[i] > 0 else "-"
        print(f"    {sign} {feature_names[i]:<25s} 값={x[0, i]:.4f}   기여도={contrib[i]:+.4f}")


def explain_dual_scores(
    model_diff,
    model_df,
    x_row: np.ndarray,
    video_name: str,
    true_label: str,
    tau_diff: float,
    tau_df: float,
    allow_ambiguous: bool = True,
    explainer_diff=None,
    explainer_df=None,
    feature_names=None,
):
    x = x_row.reshape(1, -1)
    p_diff = float(model_diff.predict_proba(x)[0, 1])
    p_df = float(model_df.predict_proba(x)[0, 1])
    pred = combine_scores(
        np.array([p_diff]), np.array([p_df]), tau_diff, tau_df,
        allow_ambiguous=allow_ambiguous,
    )[0]

    print(f"\n--- {video_name} (실제 라벨: {true_label}) ---")
    print(f"  [A real vs diffusion]  diffusion {p_diff*100:.1f}%  |  real {(1-p_diff)*100:.1f}%")
    print(f"  [B real vs deepfake]   deepfake  {p_df*100:.1f}%  |  real {(1-p_df)*100:.1f}%")
    print(f"  합친 라벨 (τ_diff={tau_diff}, τ_df={tau_df}): {pred}")

    if explainer_diff is not None and feature_names is not None:
        explain_single_binary(
            model_diff, explainer_diff, x_row, feature_names,
            positive_name="diffusion", tag="A SHAP",
        )
    if explainer_df is not None and feature_names is not None:
        explain_single_binary(
            model_df, explainer_df, x_row, feature_names,
            positive_name="deepfake", tag="B SHAP",
        )


def main():
    parser = argparse.ArgumentParser(
        description="이중 이진 분류기 (real-vs-diffusion + real-vs-deepfake)"
    )
    parser.add_argument("--csv", required=True, help="입력 CSV 경로")
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR),
                        help="진짜 영상 폴더")
    parser.add_argument("--remove-outliers", action="store_true",
                        help="IQR 이상치 제거")
    parser.add_argument("--model-out", default="xgb_model_dual_binary.joblib",
                        help="두 모델 번들 저장 경로")
    parser.add_argument("--scores-out", default="dual_binary_oof_scores.csv",
                        help="영상별 out-of-fold 점수 CSV")
    parser.add_argument("--threshold-out", default="threshold_comparison.txt",
                        help="임계값 비교표 txt 저장 경로")
    parser.add_argument("--tau-diff", type=float, default=None,
                        help="diffusion 임계값 (미지정 시 그리드 서치 추천값 자동 사용)")
    parser.add_argument("--tau-df", type=float, default=None,
                        help="deepfake 임계값 (미지정 시 그리드 서치 추천값 자동 사용)")
    parser.add_argument("--no-ambiguous", action="store_true",
                        help="둘 다 임계값 초과 시 더 높은 쪽으로 강제 결정 (ambiguous 없음)")
    parser.add_argument("--explain-n", type=int, default=5,
                        help="예시로 점수 출력할 영상 개수")
    parser.add_argument(
        "--log-out",
        default=None,
        help="학습 로그 txt 경로 (기본: train_classifier3_YYYYMMDD_HHMMSS.txt)",
    )
    args = parser.parse_args()

    log_path = args.log_out or f"train_classifier3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with redirect_stdout_to_file(log_path):
        _run(args)

    print(f"[로그 저장] {Path(log_path).resolve()}", file=sys.__stdout__)


def _run(args):
    print(f"[로드] {args.csv}")
    df = load_and_label(args.csv, real_dir=args.real_dir)
    print(df.groupby("group").size())

    if args.remove_outliers:
        df = remove_outliers_iqr(df, FEATURE_COLS)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["group"].values

    # 1) 각자 홈그라운드 이진 CV
    evaluate_binary_cv(X, y, positive="diffusion", pair_name="A real vs diffusion")
    evaluate_binary_cv(X, y, positive="deepfake", pair_name="B real vs deepfake")

    # 2) 전체 영상에 대한 OOF 점수 (운영과 같은 설정)
    print("\n=== 전체 데이터에 대한 out-of-fold 점수 (A·B 모두 적용) ===")
    p_diff = oof_positive_proba(X, y, positive="diffusion", train_classes=PAIR_DIFF)
    p_df = oof_positive_proba(X, y, positive="deepfake", train_classes=PAIR_DF)

    scores = df[["video_name"]].copy()
    scores["true_label"] = y
    scores["prob_diffusion"] = p_diff
    scores["prob_deepfake"] = p_df
    scores["prob_real_from_A"] = 1.0 - p_diff
    scores["prob_real_from_B"] = 1.0 - p_df
    scores.to_csv(args.scores_out, index=False, encoding="utf-8-sig")
    print(f"[저장] OOF 점수 -> {args.scores_out}")

    # 클래스별 평균 점수 (직관 확인)
    print("\n=== 실제 클래스별 평균 점수 ===")
    for g in CLASS_NAMES:
        m = y == g
        print(f"  {g:10s}  mean P(diffusion)={p_diff[m].mean():.3f}  "
              f"mean P(deepfake)={p_df[m].mean():.3f}")

    # 항상 그리드 서치 → 비교 txt 저장 → (미지정 시) 추천 τ 자동 적용
    _, best_td, best_tf = threshold_search(
        y, p_diff, p_df, out_txt=args.threshold_out,
    )
    tau_diff = args.tau_diff if args.tau_diff is not None else best_td
    tau_df = args.tau_df if args.tau_df is not None else best_tf
    if args.tau_diff is None or args.tau_df is None:
        print(f"[자동 적용] tau_diff={tau_diff:.2f}, tau_df={tau_df:.2f}")
    else:
        print(f"[수동 지정] tau_diff={tau_diff:.2f}, tau_df={tau_df:.2f}")

    allow_amb = not args.no_ambiguous
    evaluate_combined(y, p_diff, p_df, tau_diff, tau_df, allow_ambiguous=allow_amb)
    if allow_amb:
        evaluate_combined(y, p_diff, p_df, tau_diff, tau_df, allow_ambiguous=False)

    # 3) 최종 모델 학습·저장
    print("\n=== 전체 데이터로 최종 A·B 모델 학습 ===")
    model_diff = train_final_pair(X, y, positive="diffusion")
    model_df = train_final_pair(X, y, positive="deepfake")
    joblib.dump(
        {
            "model_diffusion": model_diff,
            "model_deepfake": model_df,
            "feature_cols": FEATURE_COLS,
            "pair_diffusion": list(PAIR_DIFF),
            "pair_deepfake": list(PAIR_DF),
            "tau_diff": tau_diff,
            "tau_df": tau_df,
        },
        args.model_out,
    )
    print(f"[저장 완료] 이중 모델 -> {args.model_out}")

    # 4) SHAP (각 모델은 자기 학습 분포 = 해당 쌍 subset으로 전역 설명)
    print("\n=== SHAP 설명 ===")
    X_diff = X[subset_mask(y, PAIR_DIFF)]
    X_dfake = X[subset_mask(y, PAIR_DF)]
    explainer_diff = explain_global_binary(
        model_diff, X_diff, FEATURE_COLS, title="A real vs diffusion",
    )
    explainer_df = explain_global_binary(
        model_df, X_dfake, FEATURE_COLS, title="B real vs deepfake",
    )

    print(f"\n=== 예시 영상 {args.explain_n}개 (점수 + SHAP) ===")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(args.explain_n, len(X)), replace=False)
    for idx in sample_idx:
        explain_dual_scores(
            model_diff, model_df, X[idx],
            df.iloc[idx]["video_name"], y[idx],
            tau_diff, tau_df,
            allow_ambiguous=allow_amb,
            explainer_diff=explainer_diff,
            explainer_df=explainer_df,
            feature_names=FEATURE_COLS,
        )


if __name__ == "__main__":
    main()
