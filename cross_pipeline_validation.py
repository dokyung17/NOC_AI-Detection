"""
데이터셋(서브그룹)별 일반화 검증 스크립트 -- Leave-One-Subgroup-Out
======================================================================

목적:
    무작위 5-fold 교차검증은 같은 데이터셋/파이프라인(예: FaceForensics++, faceswap,
    sora, veo, ...)이 학습·테스트에 항상 섞여 들어가기 때문에, 모델이 "진짜 조작 흔적"이
    아니라 "이 영상이 어느 데이터셋 출신인가"라는 지문(pipeline signature)을 학습해도
    정확도가 높게 나올 수 있다.

    실제 딥페이크/AI 생성영상 탐지 논문들은 이 문제를 피하기 위해 무작위 k-fold 대신
    "cross-dataset evaluation"(한 데이터셋으로 학습, 완전히 다른 데이터셋으로 테스트) 또는
    "cross-manipulation evaluation"(일부 기법으로 학습, 학습에 없던 기법으로 테스트)을
    표준적으로 사용한다.

    이 스크립트는 그 방식을 지금 가진 모든 서브그룹(생성모델별/데이터소스별)으로 일반화한
    "Leave-One-Subgroup-Out" 평가다:
        각 서브그룹(예: sora, veo, faceforensics(__), real(numeric), ...)을 딱 하나씩
        완전히 테스트셋으로 빼놓고, 나머지 전체로 학습한 뒤 그 서브그룹에 대한 정확도를 측정한다.
    이러면 "학습 때 전혀 본 적 없는 새로운 데이터셋/생성모델을 만났을 때의 성능"을
    서브그룹 단위로 전부 확인할 수 있다.

사용법:
    python cross_pipeline_validation.py --csv unified_features_20260716_011350.csv

필요 패키지:
    pip install xgboost scikit-learn pandas numpy

주의:
    이 스크립트는 train_classifier.py와 같은 폴더에 있어야 한다
    (FEATURE_COLS, load_and_label, classify_subgroup 등을 그 파일에서 그대로 가져와 쓴다).
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from train_classifier import (
    FEATURE_COLS,
    build_class_maps,
    classify_subgroup,
    decode_labels,
    encode_labels,
    load_and_label,
)

warnings.filterwarnings("ignore")

CLASS_NAMES = ["real", "diffusion", "deepfake"]


def _make_model(random_state=42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        missing=np.nan,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _fit_with_balanced_weight(model, X, y_str, class_to_idx):
    classes, counts = np.unique(y_str, return_counts=True)
    weight_map = {c: len(y_str) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weight = np.array([weight_map[v] for v in y_str])
    y_idx = encode_labels(y_str, class_to_idx)
    model.fit(X, y_idx, sample_weight=sample_weight)
    return model


def run_leave_one_subgroup_out(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    존재하는 모든 서브그룹(생성모델/데이터소스)을 하나씩 완전히 테스트셋으로 빼고,
    나머지 전체로 학습해서 그 서브그룹에 대한 정확도를 측정한다.
    """
    class_to_idx, idx_to_class = build_class_maps(CLASS_NAMES)
    subgroups = sorted(df["subgroup"].unique())

    rows = []
    for held_out in subgroups:
        test_mask = df["subgroup"] == held_out
        train_mask = ~test_mask

        X_train = df.loc[train_mask, FEATURE_COLS].values.astype(float)
        y_train = df.loc[train_mask, "group"].values
        X_test = df.loc[test_mask, FEATURE_COLS].values.astype(float)
        y_test = df.loc[test_mask, "group"].values

        # 이 서브그룹의 학습 후보(다른 클래스)가 학습셋에 최소 2개 클래스 이상 있어야 의미가 있음
        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            print(f"  [건너뜀] '{held_out}': 학습 데이터 클래스 부족 또는 테스트 표본 없음")
            continue

        model = _make_model(random_state)
        _fit_with_balanced_weight(model, X_train, y_train, class_to_idx)

        y_pred_idx = model.predict(X_test)
        y_pred = decode_labels(y_pred_idx, idx_to_class)

        acc = accuracy_score(y_test, y_pred)
        true_label = y_test[0]  # 서브그룹 안에서는 true label이 항상 동일 (real/diffusion/deepfake 중 하나)

        pred_counts = pd.Series(y_pred).value_counts(normalize=True)
        row = {
            "subgroup": held_out,
            "true_label": true_label,
            "n": len(y_test),
            "accuracy": acc,
        }
        for c in CLASS_NAMES:
            row[f"predicted_as_{c}"] = pred_counts.get(c, 0.0)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("accuracy")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="서브그룹(데이터셋/생성모델)별 Leave-One-Subgroup-Out 일반화 검증"
    )
    parser.add_argument("--csv", required=True, help="입력 CSV 경로 (unified_features_*.csv)")
    parser.add_argument("--out-csv", default="subgroup_generalization_results.csv",
                         help="결과를 저장할 CSV 경로")
    args = parser.parse_args()

    print(f"[로드] {args.csv}")
    df = load_and_label(args.csv)
    df["subgroup"] = df["video_name"].apply(classify_subgroup)

    print("\n서브그룹 구성 (전체 데이터셋):")
    print(df.groupby(["subgroup", "group"]).size())

    print("\n" + "=" * 78)
    print("Leave-One-Subgroup-Out 평가 진행 중")
    print("(각 서브그룹을 학습에서 완전히 제외하고, 그 서브그룹만으로 테스트)")
    print("=" * 78)

    result = run_leave_one_subgroup_out(df)

    print("\n=== 서브그룹별 일반화 정확도 (오름차순 = 가장 취약한 것부터) ===")
    print(result.round(3).to_string(index=False))

    result.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] -> {args.out_csv}")

    print("\n" + "=" * 78)
    print("=== 참고: 기존 무작위 5-fold 분할 정확도(약 0.76~0.78)와 비교 ===")
    print("위 표에서 값이 그보다 많이 낮은 서브그룹은,")
    print("모델이 그 서브그룹 고유의 패턴/파이프라인 지문 없이는 잘 못 맞춘다는 뜻이다.")
    print("즉 '진짜 조작 흔적'이 아니라 '그 데이터셋 특유의 신호'에 의존하고 있었을 가능성이 높다.")

    # true_label별 평균도 같이 보여주기 (real 전체 / diffusion 전체 / deepfake 전체 일반화 경향)
    print("\n=== true_label(대분류)별 평균 일반화 정확도 ===")
    print(result.groupby("true_label")["accuracy"].agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
