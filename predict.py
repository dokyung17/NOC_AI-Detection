"""
저장된 모델로 새 영상(들)을 예측하는 스크립트

사용법:
    python predict.py --model xgb_model.joblib --csv new_videos_features.csv
    python predict.py --model xgb_model_3class_0824.joblib --csv results/unified_features_20260831_210753.csv --out results/predict_test.csv

new_videos_features.csv는 train_classifier.py가 읽는 것과 같은 포맷
(video_name 컬럼 + 7개 피처 컬럼)이어야 합니다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict_to_dataframe(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    X = df[feature_cols].values.astype(float)
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)

    rows = []
    for i in range(len(df)):
        name = df.iloc[i].get("video_name", f"row_{i}")
        idx = int(pred_idx[i])
        row = {
            "video_name": name,
            "pred_class": class_names[idx],
            "pred_confidence": float(proba[i, idx]),
        }
        for j, c in enumerate(class_names):
            row[f"proba_{c}"] = float(proba[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def default_out_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.startswith("unified_features_"):
        suffix = stem.replace("unified_features_", "", 1)
        return csv_path.parent / f"predict_{suffix}.csv"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return csv_path.parent / f"predict_{ts}.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="joblib으로 저장된 모델 경로")
    parser.add_argument("--csv", required=True, help="예측할 영상들의 피처 CSV")
    parser.add_argument(
        "--out",
        default=None,
        help="예측 결과 CSV 저장 경로 (기본: results/predict_<입력파일명>.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="영상별 SHAP 설명을 터미널에 출력",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    class_names = bundle["class_names"]

    csv_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(csv_path)
    X = df[feature_cols].values.astype(float)

    out_path = Path(args.out).expanduser().resolve() if args.out else default_out_path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = predict_to_dataframe(model, df, feature_cols, class_names)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    counts = result_df["pred_class"].value_counts()
    print(f"Saved {len(result_df)} predictions → {out_path}")
    print("예측 분포:")
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt}")

    if args.verbose:
        import shap
        from train_classifier import explain_single_video

        explainer = shap.TreeExplainer(model)
        for i in range(len(df)):
            name = df.iloc[i].get("video_name", f"row_{i}")
            print(f"\n===================== {name} =====================")
            explain_single_video(model, explainer, X[i], feature_cols, class_names)


if __name__ == "__main__":
    main()
