# train_classifier2 Experiment Results

생성 시각: 2026-09-13 23:31:05

## 1. Experiment Setup

- Task: Real / Deepfake / Diffusion 3-class Classification
- Dataset: `results/unified_features_20260911_024351.csv`
- Model: XGBoost (multi:softprob, n_estimators=300, max_depth=4, lr=0.05)
- Class Weight: balanced sample weight
- Features: 고정 10개 (select_features.py 선별 결과)
- Validation: train 내부 5-Fold Stratified CV
- Test Set: 최종 평가에만 1회 사용
- Total Samples: 800
- Train / Test: 640 / 160

클래스별 샘플 수

| Class | Total | Train | Test |
|---|---:|---:|---:|
| Real | 335 | 268 | 67 |
| Deepfake | 248 | 198 | 50 |
| Diffusion | 217 | 174 | 43 |

## 2. Features

- absdiff_std
- bvp_std
- d3_temporal_score
- highfreq_score
- patch_corr_mean
- patch_signal_std_mean
- boundary_score_mean
- boundary_score_std
- boundary_score_max
- identity_sim_std

**총 feature 수: 10개**

## 3. Train CV Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7625 |
| Macro Precision | 0.7614 |
| Macro Recall | 0.7683 |
| Macro F1 | 0.7643 |
| Weighted Precision | 0.7631 |
| Weighted Recall | 0.7625 |
| Weighted F1 | 0.7622 |
| Macro AUC | 0.8991 |
| Weighted AUC | 0.8942 |

## 4. Final Model Performance (test set)

| Metric | Score |
|---|---:|
| Accuracy | 0.7875 |
| Macro Precision | 0.7940 |
| Macro Recall | 0.7902 |
| Macro F1 | 0.7894 |
| Weighted Precision | 0.7905 |
| Weighted Recall | 0.7875 |
| Weighted F1 | 0.7866 |
| Macro AUC | 0.9058 |
| Weighted AUC | 0.9011 |

## 5. Per-Class Performance

| class | precision | recall | f1_score | support |
|---|---:|---:|---:|---:|
| Real | 0.7812 | 0.7463 | 0.7634 | 67 |
| Deepfake | 0.7586 | 0.8800 | 0.8148 | 50 |
| Diffusion | 0.8421 | 0.7442 | 0.7901 | 43 |
| Macro Avg | 0.7940 | 0.7902 | 0.7894 | 160 |
| Weighted Avg | 0.7905 | 0.7875 | 0.7866 | 160 |

```text
              precision    recall  f1-score   support

        Real      0.781     0.746     0.763        67
    Deepfake      0.759     0.880     0.815        50
   Diffusion      0.842     0.744     0.790        43

    accuracy                          0.787       160
   macro avg      0.794     0.790     0.789       160
weighted avg      0.791     0.787     0.787       160
```

## 6. Confusion Matrix

| Actual \ Predicted | Real | Deepfake | Diffusion |
|---|---:|---:|---:|
| Real | 50 | 11 | 6 |
| Deepfake | 6 | 44 | 0 |
| Diffusion | 8 | 3 | 32 |

![Confusion Matrix](confusion_matrix.png)

## 7. SHAP Feature Importance

| rank | feature | mean_abs_shap |
|---|---:|---:|
| 1 | bvp_std | 0.6978 |
| 2 | identity_sim_std | 0.6016 |
| 3 | boundary_score_std | 0.3071 |
| 4 | highfreq_score | 0.2778 |
| 5 | patch_signal_std_mean | 0.2368 |
| 6 | boundary_score_mean | 0.2311 |
| 7 | d3_temporal_score | 0.1926 |
| 8 | boundary_score_max | 0.1776 |
| 9 | patch_corr_mean | 0.1292 |
| 10 | absdiff_std | 0.0943 |

![SHAP Summary](shap_summary.png)

### 클래스별 mean |SHAP|

| feature | mean_abs_shap_real | mean_abs_shap_diffusion | mean_abs_shap_deepfake | mean_abs_shap_overall |
|---|---:|---:|---:|---:|
| bvp_std | 0.4771 | 1.4394 | 0.1770 | 0.6978 |
| identity_sim_std | 0.7628 | 0.1726 | 0.8695 | 0.6016 |
| boundary_score_std | 0.2064 | 0.5899 | 0.1250 | 0.3071 |
| highfreq_score | 0.2677 | 0.2526 | 0.3129 | 0.2778 |
| patch_signal_std_mean | 0.1898 | 0.2326 | 0.2880 | 0.2368 |
| boundary_score_mean | 0.1594 | 0.1215 | 0.4124 | 0.2311 |
| d3_temporal_score | 0.1895 | 0.2074 | 0.1808 | 0.1926 |
| boundary_score_max | 0.1449 | 0.2187 | 0.1694 | 0.1776 |
| patch_corr_mean | 0.1119 | 0.1236 | 0.1521 | 0.1292 |
| absdiff_std | 0.0637 | 0.0876 | 0.1315 | 0.0943 |

![SHAP real](shap_summary_real.png)
![SHAP diffusion](shap_summary_diffusion.png)
![SHAP deepfake](shap_summary_deepfake.png)

## 8. Summary

- 고정 feature 10개로 train 640 / test 160 를 사용했다.
- train 내부 CV Macro F1 은 0.7643, Macro AUC 는 0.8991 였다.
- test Accuracy 는 0.7875, Macro F1 은 0.7894, Macro AUC 는 0.9058 를 기록했다.
- SHAP 에서 가장 영향력이 높은 feature 는 `bvp_std` (mean |SHAP| = 0.6978) 였다.
