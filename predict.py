"""
저장된 모델로 새 영상(들)을 예측하는 스크립트

사용법:
    python predict.py --model xgb_model.joblib --csv new_videos_features.csv

new_videos_features.csv는 train_classifier.py가 읽는 것과 같은 포맷
(video_name 컬럼 + 7개 피처 컬럼)이어야 합니다.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import shap

from train_classifier import explain_single_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="joblib으로 저장된 모델 경로")
    parser.add_argument("--csv", required=True, help="예측할 영상들의 피처 CSV")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model, feature_cols, class_names = bundle["model"], bundle["feature_cols"], bundle["class_names"]

    df = pd.read_csv(args.csv)
    X = df[feature_cols].values.astype(float)

    explainer = shap.TreeExplainer(model)

    for i in range(len(df)):
        print(f"\n===================== {df.iloc[i].get('video_name', f'row_{i}')} =====================")
        explain_single_video(model, explainer, X[i], feature_cols, class_names)


if __name__ == "__main__":
    main()
