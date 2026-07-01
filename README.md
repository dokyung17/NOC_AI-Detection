# NOC_AI-Detection

AI 생성·조작 영상 탐지를 위한 **단일 통합 프로젝트**입니다.  
기존 `D3/`, `pyVHR/`에서 **실제로 쓰는 코드만** 가져와 정리했습니다.

## 시작하기

### 1. 요구 사항

- **Python 3.9 이상** (3.10~3.12 권장)
- macOS / Linux / Windows
- 디스크 여유 약 2GB 이상 (PyTorch + 패키지 + 영상)
- GPU는 선택 사항 (`--device cpu`가 기본값)

### 2. 프로젝트 폴더로 이동

```bash
cd /Users/MacBook/Dropbox/Mac/Desktop/NOC_AI-Detection
```

다른 PC로 옮긴 경우, 위 경로를 본인 환경의 `NOC_AI-Detection` 폴더 경로로 바꿉니다.

### 3. 가상환경 만들기 (권장)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 4. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

첫 실행 시 D3 인코더(ResNet-18 등) **가중치가 자동 다운로드**될 수 있습니다. 인터넷 연결이 필요합니다.

### 5. 동작 확인 (영상 1개)

```bash
python run.py data/subject1.mp4
```

정상이면 터미널에 메트릭이 출력되고, `results/unified_features_YYYYMMDD_HHMMSS.csv` 파일이 생성됩니다.

### 6. 전체 데이터셋 실행

```bash
python run.py
```

인자 없이 실행하면 `data/videos/` 아래 **real 15개 + fake 20개**를 순서대로 처리합니다.  
35개 전체는 CPU 기준 **수십 분~1시간 이상** 걸릴 수 있습니다.

### 7. 결과 확인

```bash
ls results/
```

CSV를 열면 영상별로 rPPG 5개 + D3 2개 특징과 `label`(real/fake)이 들어 있습니다.

### 자주 나는 문제


| 증상                        | 해결                                                 |
| ------------------------- | -------------------------------------------------- |
| `No module named 'torch'` | 가상환경 활성화 후 `pip install -r requirements.txt` 다시 실행 |
| `No videos found`         | `data/videos/`에 mp4가 있는지, 경로가 맞는지 확인               |
| MediaPipe / OpenCV 오류     | `pip install mediapipe opencv-python` 재설치          |
| 얼굴 미검출로 실패                | 정면 얼굴이 나오는 영상인지 확인 (`success=False` 행 참고)          |


---

## 하는 일

비디오 1개마다 **rPPG 5개 + D3 2개** 특징을 추출해 CSV로 저장합니다.

```
mp4
 ├─→ pyVHR/     → features/rppg_metrics  →  rPPG 5메트릭
 └─→ visual/    → d3_model + highfreq    →  d3_temporal_score, highfreq_score
                                                      ↓
                                              results/*.csv
```

## 폴더 설명


| 폴더              | 출처         | 역할                                 |
| --------------- | ---------- | ---------------------------------- |
| `**pyVHR/**`    | 원본 `pyVHR` | **rPPG 엔진** — 얼굴 피부에서 맥동 신호 추출     |
| `**visual/`**   | 원본 `D3`    | **시각 특징** — 시간(D3) + 공간(Laplacian) |
| `**features/`** | 이 프로젝트 전용  | **통합 유틸** — 영상 입출력, rPPG 5메트릭 계산   |
| `**run.py`**    | —          | 위 모듈을 연결하는 **메인 실행 파일**            |


### `pyVHR/` — 생체신호(rPPG)

원본 pyVHR에서 실제로 쓰는 부분만 잘라 둔 폴더입니다.

```
pyVHR/
├── analysis/pipeline.py   # run.py가 호출 (ConvexHull + patches + POS + Welch)
├── extraction/            # MediaPipe 얼굴 검출, 피부 ROI, 패치 RGB 추출
├── BVP/                   # POS: RGB → BVP(맥동 파형)
└── BPM/                   # Welch: BVP → BPM
```

**흐름:** mp4 → 얼굴 검출 → 피부 패치 → POS → BVP → Welch

### `visual/` — D3 기반 시각 특징

원본 D3에서 모델·Laplacian·프레임 로딩만 가져온 폴더입니다.

```
visual/
├── d3_model.py    # Vision Encoder → 연속 프레임 특징 변화 → d3_temporal_score
├── highfreq.py    # Laplacian 분산 → highfreq_score
└── frames.py      # mp4에서 프레임 로드·전처리
```

- `d3_temporal_score` — 프레임 간 특징 불연속성 (**시간**)
- `highfreq_score` — 프레임 내 에지·질감 강도 (**공간**)

rPPG와 **별개 경로**로, 같은 mp4를 따로 읽어 처리합니다.

### `features/` — 통합 전용 코드

D3·pyVHR 원본에는 없고, 이 프로젝트를 만들면서 추가한 폴더입니다.

```
features/
├── io.py            # 비디오 목록 탐색, real/fake 라벨 부여
└── rppg_metrics.py  # BVP·패치 신호 → rPPG 5메트릭 계산
```

`pyVHR/`가 신호를 **뽑고**, `features/`가 탐지용 **숫자로 변환**합니다.

## 제외된 것 (원본에 있었으나 미사용)

- pyVHR: GUI, deepRPPG, datasets, notebooks, faceparsing, OMIT/CHROM 등 다른 rPPG 방법
- D3: `eval.py` AP 배치 평가 (데이터셋 전체 AP 계산)

### `video2frame` / `folder2csv`는 안 쓰나요?

**이 프로젝트에서는 사용하지 않습니다.** 역할은 `visual/frames.py`가 대신합니다.

원본 D3는 **3단계 전처리** 후 평가했습니다.

```
[원본 D3]
mp4 → video2frame.py  → frames/*.jpg 저장 (3초 구간, 8fps)
    → folder2csv.py   → real.csv / fake.csv 생성
    → eval.py         → CSV의 프레임 폴더 읽어 D3 + HighFreq AP 계산
```

이 프로젝트는 **mp4를 실행 시 바로** 읽습니다.

```
[NOC_AI-Detection]
mp4 → visual/frames.py  → 메모리에서 프레임 로드·전처리
    → visual/d3_model.py + highfreq.py
```


|     | 원본 D3                        | 이 프로젝트                      |
| --- | ---------------------------- | --------------------------- |
| 입력  | 디스크에 저장된 jpg 프레임 + CSV       | **mp4 직접**                  |
| 전처리 | `video2frame` + `folder2csv` | `visual/frames.py`          |
| 라벨  | CSV (`folder2csv`)           | `data/videos/real|fake` 폴더명 |
| 평가  | `eval.py` (AP 일괄 계산)         | `run.py` (영상별 특징 CSV)       |


**전처리 규칙**(224 리사이즈, center crop 10%, ImageNet normalize)은 D3 `datasets.py`와 비슷하게 맞춰 두었지만, **프레임 뽑는 방식은 다릅니다.**

- `video2frame`: 영상 **3초 구간만** 8fps로 잘라 jpg 저장
- `visual/frames.py`: 영상 **전체**에서 8~16장을 **균등 샘플링**

그래서 원본 D3 `eval.py`의 AP 수치와 **완전히 같지는 않을 수** 있습니다.  
논문·발표에서 원본 D3 AP(0.94 등)를 인용할 때는 `Desktop/D3/`의 `video2frame` → `folder2csv` → `eval.py` 흐름 결과를 따로 참고하세요.

## 사용법 (옵션)

```bash
# 기본: data/videos 전체 (35개)
python run.py

# 단일 비디오
python run.py data/videos/fake/subject13_ai.mp4

# 다른 폴더 지정
python run.py /path/to/videos/ --pattern "*.mp4"

# real/fake 폴더 없이 한 폴더에 모아 둔 경우
python run.py /path/to/videos/ --real-stems subject1,subject11

# D3 인코더 변경 (기본: ResNet-18)
python run.py data/videos/real/subject1.mp4 --encoder XCLIP-16

# GPU 사용 (CUDA 가능한 환경)
python run.py --device cuda
```


| 옵션                               | 설명                                | 기본값           |
| -------------------------------- | --------------------------------- | ------------- |
| `inputs`                         | 비디오 파일 또는 폴더 (생략 시 `data/videos`) | `data/videos` |
| `--pattern`                      | 폴더 검색 시 glob                      | `*.mp4`       |
| `--recursive` / `--no-recursive` | 하위 폴더까지 검색                        | `--recursive` |
| `--save-dir`                     | CSV 저장 폴더                         | `results/`    |
| `--real-stems`                   | real로 쓸 파일명 stem (쉼표 구분)          | 폴더명으로 자동      |
| `--encoder`                      | D3 vision encoder                 | `ResNet-18`   |
| `--device`                       | `cpu` 또는 `cuda`                   | `cpu`         |


## 데이터

영상은 `data/videos/real/` (15개), `data/videos/fake/` (20개)에 있습니다.

`real` / `fake` 폴더는 **처리 방식을 바꾸지 않습니다.**  
같은 파이프라인으로 돌리고, 결과 CSV의 `label` 컬럼만 자동으로 붙이기 위한 구분입니다.  
한 폴더에 모아도 되며, 그때는 `--real-stems`로 라벨을 지정할 수 있습니다.

## 출력 CSV 컬럼


| 구분   | 컬럼                                                                                    |
| ---- | ------------------------------------------------------------------------------------- |
| 메타   | `video_name`, `label`, `success`                                        |
| rPPG | `absdiff_std`, `bvp_std`, `patch_corr_mean`, `patch_signal_std_mean`, `pseudo_snr_db` |
| D3   | `d3_temporal_score`, `highfreq_score`                                                 |


## 프로젝트 구조

```
NOC_AI-Detection/
├── run.py                 # 메인 진입점
├── data/videos/
│   ├── real/              # 15개
│   └── fake/              # 20개
├── pyVHR/                 # rPPG (원본 pyVHR slim)
├── visual/                # D3 + Laplacian (원본 D3)
├── features/              # 입출력 + rPPG 메트릭 (통합 전용)
└── results/               # 실행 결과 CSV
```

## 원본 프로젝트와의 관계

- `Desktop/D3/`, `Desktop/pyVHR/`는 **백업·재현용**으로 남겨 두고, **이후 작업은 이 폴더만** 사용하면 됩니다.
- 영상은 이 프로젝트의 `data/videos/`로 옮겨 두었습니다.

