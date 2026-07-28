# NOC_AI-Detection

AI 생성·조작 영상 탐지를 위한 **단일 통합 프로젝트**입니다.

기존 `D3/`, `pyVHR/` 프로젝트에서 실제 탐지에 사용하는 코드만 가져와 하나의 실행 파이프라인으로 정리했습니다.

각 영상에서 다음 특징을 추출합니다.

- rPPG 기반 생체신호 특징 5개
- D3 기반 시간적 특징 1개
- Laplacian 기반 공간적 특징 1개

추출된 특징은 영상별 CSV 파일로 저장됩니다.

---

## 주요 업데이트: GPU 가속 · 다중 프로세스 병렬 처리

기존 CPU 중심 실행 방식에 `--device cuda` 옵션을 추가하여, CUDA를 지원하는 NVIDIA GPU에서 **D3 시각 특징 추출을 가속**할 수 있도록 수정했습니다. 또한 `--workers` 옵션으로 **여러 영상을 동시에 처리**할 수 있습니다.

### 변경 내용

- `--device` 실행 옵션 추가
- `cpu`와 `cuda` 장치 선택 지원 (`auto`는 CUDA 사용 가능 시 자동 선택)
- D3 모델을 지정한 장치로 이동
- 영상 프레임 텐서를 GPU로 이동하여 D3 추론 수행
- `torch.inference_mode()`를 사용하여 추론 과정의 불필요한 그래디언트 계산 방지
- GPU 사용 시 D3 시간 특징과 고주파 특징 계산 속도 개선
- `--workers`로 `ProcessPoolExecutor` 기반 영상 단위 병렬 처리 지원
- 기존 CPU 단일 워커 실행 방식 유지 (`--workers 1`)

### GPU가 적용되는 범위

현재 GPU 가속은 주로 다음 시각 특징 추출 과정에 적용됩니다.

```text
영상
 └─→ visual/frames.py
      └─→ 프레임 Tensor를 GPU로 이동
           ├─→ D3_model
           └─→ Laplacian High-Frequency
```

rPPG 과정은 현재 `cpu_POS` 방식을 사용하므로 CPU에서 실행됩니다.

```text
영상
 └─→ pyVHR
      └─→ 얼굴 검출 → 피부 패치 → POS → BVP → Welch
```

따라서 `--device cuda`를 사용하더라도 전체 처리 과정이 모두 GPU에서 실행되는 것은 아닙니다. D3 기반 시각 특징 추출 부분이 GPU로 가속됩니다.

### 병렬 처리 (`--workers`)

영상 여러 개를 처리할 때 `--workers N`으로 워커 프로세스 N개를 띄워 동시에 추출할 수 있습니다. 병목인 rPPG(MediaPipe·패치·POS)가 CPU에서 돌아가므로, 코어가 남는 환경에서는 GPU만 쓰는 것보다 전체 처리 시간이 더 줄어드는 경우가 많습니다.

```text
워커 1 ──→ 영상 A (rPPG CPU + D3 device)
워커 2 ──→ 영상 B (rPPG CPU + D3 device)
워커 N ──→ 영상 … 
              ↓
         results/*.csv  (입력 영상 순서 유지)
```

- 기본값: `--workers 1` (기존과 동일한 순차 처리)
- 권장: CPU 코어·RAM에 맞게 `2`~`4`부터 시작, 여유 있으면 `6`~`8`까지 올려 보기
- 워커마다 Pipeline과 D3 모델을 따로 로드합니다. `--device cuda`일 때 **VRAM 사용량은 워커 수에 비례**합니다.
- 특징 추출 알고리즘·결과 의미는 동일하며, 완료 로그 출력 순서만 달라질 수 있습니다. CSV 행 순서는 입력 영상 순서를 유지합니다.

---



## 처리 흐름

비디오 한 개마다 **rPPG 5개 + D3 계열 2개** 특징을 추출하여 CSV로 저장합니다.

```text
mp4
 ├─→ pyVHR/
 │    └─→ features/rppg_metrics.py
 │         └─→ rPPG 특징 5개
 │
 └─→ visual/
      ├─→ d3_model.py
      │    └─→ d3_temporal_score
      └─→ highfreq.py
           └─→ highfreq_score
                    ↓
             results/*.csv
```

---



# 시작하기



## 1. 요구 사항



### 공통 환경

- Python 3.9 이상
- Python 3.10~3.12 권장
- macOS / Linux / Windows
- 디스크 여유 공간 약 2GB 이상
- 인터넷 연결
  - 첫 실행 시 D3 인코더 가중치가 다운로드될 수 있음



### CPU 실행

별도의 GPU 없이 실행할 수 있습니다.

```text
기본 장치: cpu
```



### GPU 실행

GPU 가속을 사용하려면 다음 환경이 필요합니다.

- CUDA를 지원하는 NVIDIA GPU
- GPU 드라이버
- CUDA를 지원하는 PyTorch 설치
- 충분한 GPU 메모리

> Apple Silicon의 MPS와 AMD ROCm은 현재 `--device` 사용 흐름에서 별도로 검증되지 않았습니다. 현재 README는 NVIDIA CUDA 사용을 기준으로 설명합니다.

---



## 2. 저장소 내려받기

```bash
git clone https://github.com/dokyung17/NOC_AI-Detection.git
cd NOC_AI-Detection
```

이미 저장소를 내려받았다면 프로젝트 폴더로 이동합니다.

```bash
cd /path/to/NOC_AI-Detection
```

기존 로컬 환경에서는 다음과 같은 절대 경로를 사용할 수도 있습니다.

```bash
cd /Users/MacBook/Dropbox/Mac/Desktop/NOC_AI-Detection
```

다른 PC에서는 본인의 실제 프로젝트 경로로 변경해야 합니다.

---



## 3. 가상환경 만들기

가상환경 사용을 권장합니다.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```



### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```



### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

가상환경이 정상적으로 활성화되면 터미널 앞에 다음과 같이 표시됩니다.

```text
(.venv)
```

---



## 4. 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

주요 패키지는 다음과 같습니다.

- PyTorch
- torchvision
- OpenCV
- MediaPipe
- NumPy
- SciPy
- pandas
- timm
- transformers
- albumentations

첫 실행 시 ResNet-18, XCLIP 등 선택한 D3 인코더의 사전학습 가중치가 자동으로 다운로드될 수 있습니다.

---



# CPU 실행 방법



## 단일 영상 실행

```bash
python run.py data/subject1.mp4
```

CPU 장치를 명시하려면 다음과 같이 실행합니다.

```bash
python run.py data/subject1.mp4 --device cpu
```



## 전체 데이터셋 실행

```bash
python run.py
```

인자를 생략하면 기본적으로 다음 폴더를 탐색합니다.

```text
data/videos/
├── real/
└── fake/
```

CPU 환경에서는 전체 처리에 수십 분에서 1시간 이상 걸릴 수 있습니다. 처리 시간은 CPU 성능, 영상 길이, 해상도, 얼굴 검출 상태에 따라 달라집니다.

영상 개수가 많을 때는 병렬 워커를 함께 쓰는 것을 권장합니다.

```bash
python run.py --device cpu --workers 4
```

---



# GPU 실행 방법



## 1. CUDA 사용 가능 여부 확인

다음 명령을 실행합니다.

```bash
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda); print('GPU 개수:', torch.cuda.device_count())"
```

CUDA를 정상적으로 사용할 수 있다면 다음과 비슷하게 출력됩니다.

```text
CUDA 사용 가능: True
CUDA 버전: 12.x
GPU 개수: 1
```

GPU 이름을 확인하려면 다음 명령을 사용합니다.

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA GPU 없음')"
```

예시:

```text
NVIDIA GeForce RTX 4070
```



## 2. 단일 영상 GPU 실행

```bash
python run.py data/subject1.mp4 --device cuda
```



## 3. 전체 데이터셋 GPU 실행

```bash
python run.py --device cuda
```

병렬 워커와 함께 실행하려면 다음과 같이 합니다.

```bash
python run.py --device cuda --workers 4
```

Windows PowerShell 예시:

```powershell
python run.py --device cuda --workers 4
```



## 4. 특정 인코더와 GPU 함께 사용

```bash
python run.py data/videos/fake/subject13_ai.mp4 \
  --encoder XCLIP-16 \
  --device cuda
```

Windows PowerShell에서는 다음처럼 한 줄로 실행할 수 있습니다.

```powershell
python run.py data/videos/fake/subject13_ai.mp4 --encoder XCLIP-16 --device cuda
```

---



## GPU 추론 속도 개선

GPU 사용의 주요 목적은 D3 기반 시각 특징 추출 속도를 개선하는 것입니다.

### CPU 실행

```bash
python run.py --device cpu
```



### GPU 실행

```bash
python run.py --device cuda
```

GPU 환경에서는 다음 과정이 가속됩니다.

- 영상 프레임 텐서 연산
- Vision Encoder 추론
- 연속 프레임 특징 변화 계산
- D3 시간 불연속성 특징 계산
- Laplacian 기반 고주파 특징 계산

다만 다음 작업은 CPU 처리 비중이 높습니다.

- 영상 디코딩
- MediaPipe 얼굴 검출
- 피부 ROI 및 패치 추출
- POS 기반 BVP 계산
- Welch 기반 주파수 분석
- CSV 저장

따라서 실제 전체 속도 향상 폭은 다음 조건에 따라 달라집니다.

- GPU 모델
- CPU 성능
- 영상 길이와 해상도
- D3 인코더 종류
- 처리할 영상 개수
- 프레임 샘플링 수
- 얼굴 검출 및 rPPG 처리 시간
- `--workers` 값 (병렬 영상 수)
---



## 다중 프로세스 병렬 처리

`run.py`는 `--workers`로 영상 단위 병렬 처리를 지원합니다.

```bash
# 순차 처리 (기본)
python run.py --device cuda --workers 1

# 4개 영상 동시 처리
python run.py --device cuda --workers 4
```

### 동작 방식

- `ProcessPoolExecutor`로 워커 프로세스를 생성합니다.
- 각 워커는 시작 시 Pipeline과 D3 모델을 한 번 로드한 뒤, 할당된 영상을 처리합니다.
- 결과 CSV의 행 순서는 입력으로 찾은 영상 순서를 유지합니다.
- 터미널 진행 로그(`[완료수/전체]`)는 끝나는 순서대로 출력될 수 있습니다.

### 워커 수 선택 가이드

| 조건 | 권장 |
| ---- | ---- |
| 기본 / 안정 | `2`~`4` |
| CPU 코어가 많고 RAM 여유 | `6`~`8` |
| 논리 코어 수 이상 | 이득이 거의 없음 |
| `XCLIP` 등 무거운 인코더 + CUDA | VRAM을 보고 `2`~`4` |
| `CUDA out of memory` 발생 | `--workers`를 낮추거나 가벼운 인코더 사용 |

단일 영상만 처리할 때는 `--workers`를 올려도 이득이 없습니다. 데이터셋 일괄 실행에 사용하세요.

---



# 실행 결과 확인

정상적으로 실행되면 터미널에 영상별 처리 결과가 출력됩니다.

예시:

```text
[1/35] subject1.mp4
  rPPG pseudo_snr=... | D3=... | HF=...

Done 35/35 → results/unified_features_YYYYMMDD_HHMMSS.csv
```

결과 파일은 기본적으로 `results/` 폴더에 저장됩니다.

```bash
ls results/
```

Windows에서는 다음 명령을 사용할 수 있습니다.

```cmd
dir results
```

결과 파일명 형식은 다음과 같습니다.

```text
unified_features_YYYYMMDD_HHMMSS.csv
```

예시:

```text
unified_features_20260726_153020.csv
```

---



# 사용법



## 기본 실행

```bash
python run.py
```



## 단일 비디오 실행

```bash
python run.py data/videos/fake/subject13_ai.mp4
```



## 여러 영상 직접 지정

```bash
python run.py \
  data/videos/real/subject1.mp4 \
  data/videos/fake/subject13_ai.mp4
```



## 다른 폴더 지정

```bash
python run.py /path/to/videos/ --pattern "*.mp4"
```



## 하위 폴더를 포함하지 않고 검색

```bash
python run.py /path/to/videos/ --no-recursive
```



## CSV 저장 폴더 변경

```bash
python run.py --save-dir outputs/
```



## real/fake 폴더 없이 한 폴더에 모아 둔 경우

```bash
python run.py /path/to/videos/ \
  --real-stems subject1,subject11
```



## D3 인코더 변경

기본 인코더는 `ResNet-18`입니다.

```bash
python run.py data/videos/real/subject1.mp4 \
  --encoder XCLIP-16
```



## D3 특징 거리 계산 방식 변경

기본값은 `l2`입니다.

```bash
python run.py data/videos/real/subject1.mp4 \
  --loss cos
```



## GPU 실행

```bash
python run.py --device cuda
```



## 병렬 처리

```bash
python run.py --device cuda --workers 4
```



## GPU와 다른 저장 위치 함께 사용

```bash
python run.py \
  --device cuda \
  --workers 4 \
  --save-dir results_gpu/
```

---



## 실행 옵션


| 옵션               | 설명                                      | 기본값           |
| ---------------- | --------------------------------------- | ------------- |
| `inputs`         | 비디오 파일 또는 폴더 경로                         | `data/videos` |
| `--pattern`      | 폴더에서 검색할 파일 패턴                          | `*.mp4`       |
| `--recursive`    | 하위 폴더까지 검색                              | 활성화           |
| `--no-recursive` | 현재 폴더만 검색                               | 비활성화          |
| `--save-dir`     | 결과 CSV 저장 폴더                            | `results/`    |
| `--real-stems`   | real로 지정할 파일 stem 목록                    | 폴더명으로 자동 판별   |
| `--encoder`      | D3 Vision Encoder                       | `ResNet-18`   |
| `--loss`         | D3 특징 변화 계산 방식: `l2`, `cos`             | `l2`          |
| `--device`       | PyTorch 실행 장치: `auto`, `cpu`, `cuda`   | `auto`        |
| `--workers`      | 병렬 워커 프로세스 수 (영상 단위, `1`이면 순차)          | `1`           |


전체 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python run.py --help
```

---



# 하는 일

영상 한 개를 입력하면 두 개의 독립적인 경로로 처리합니다.

## 1. rPPG 생체신호 경로

```text
mp4
 → 얼굴 검출
 → 피부 영역 추출
 → 패치별 RGB 신호 추출
 → POS
 → BVP 파형
 → Welch 주파수 분석
 → rPPG 특징 5개
```

추출되는 특징:

- `absdiff_std`
- `bvp_std`
- `patch_corr_mean`
- `patch_signal_std_mean`
- `pseudo_snr_db`



## 2. D3 시각 특징 경로

```text
mp4
 → 프레임 균등 샘플링
 → 이미지 전처리
 → Vision Encoder
 → 연속 프레임 특징 변화
 → 시간적 불연속성 계산
 → Laplacian 기반 고주파 특징 계산
```

추출되는 특징:

- `d3_temporal_score`
- `highfreq_score`

rPPG와 D3는 같은 영상을 입력으로 사용하지만 서로 별도의 처리 경로로 실행됩니다.

---



# 폴더 설명


| 폴더 또는 파일       | 출처       | 역할                       |
| -------------- | -------- | ------------------------ |
| `pyVHR/`       | 원본 pyVHR | 얼굴 피부에서 rPPG 생체신호 추출     |
| `visual/`      | 원본 D3    | 시간적 특징 및 공간적 고주파 특징 추출   |
| `features/`    | 프로젝트 전용  | 영상 탐색, 라벨 처리, rPPG 특징 계산 |
| `run.py`       | 프로젝트 전용  | 모든 모듈을 연결하는 메인 실행 파일     |
| `data/videos/` | 데이터      | real/fake 영상 저장          |
| `results/`     | 실행 결과    | 영상별 특징 CSV 저장            |


---



## `pyVHR/` — 생체신호 기반 특징

원본 pyVHR에서 현재 프로젝트가 실제로 사용하는 부분만 정리한 폴더입니다.

```text
pyVHR/
├── analysis/
│   └── pipeline.py
├── extraction/
├── BVP/
└── BPM/
```



### 주요 역할

```text
analysis/pipeline.py
 └─→ run.py에서 호출
      ├─→ ConvexHull
      ├─→ 피부 패치 추출
      ├─→ POS
      └─→ Welch
```

세부 흐름:

```text
mp4
 → MediaPipe 얼굴 검출
 → 피부 ROI
 → 패치 RGB
 → POS
 → BVP
 → Welch
```

현재 `run.py`에서는 다음 방식으로 실행됩니다.

```text
roi_approach = patches
method       = cpu_POS
bpm_type     = welch
post_filt    = True
```

---



## `visual/` — D3 기반 시각 특징

원본 D3에서 모델, 프레임 처리, Laplacian 특징 계산에 필요한 코드를 가져온 폴더입니다.

```text
visual/
├── d3_model.py
├── highfreq.py
└── frames.py
```



### 파일별 역할

- `d3_model.py`
  - Vision Encoder 실행
  - 연속 프레임 특징 변화 계산
  - `d3_temporal_score` 생성
- `highfreq.py`
  - Laplacian 응답 계산
  - 영상의 에지 및 질감 강도 측정
  - `highfreq_score` 생성
- `frames.py`
  - mp4 영상 직접 읽기
  - 프레임 샘플링
  - 이미지 크기 조정 및 정규화
  - 모델 입력 텐서 생성



### 추출 특징



#### `d3_temporal_score`

연속된 영상 프레임의 시각 특징 변화와 시간적 불연속성을 나타냅니다.

#### `highfreq_score`

각 프레임의 에지, 질감 및 고주파 성분 강도를 나타냅니다.

---



## `features/` — 통합 프로젝트 전용 코드

D3와 pyVHR 원본에는 없으며, 두 파이프라인을 하나로 통합하면서 추가한 코드입니다.

```text
features/
├── io.py
└── rppg_metrics.py
```



### `features/io.py`

- 입력 영상 파일 탐색
- 폴더 재귀 검색
- real/fake 라벨 추론
- `--real-stems` 옵션 처리



### `features/rppg_metrics.py`

- pyVHR가 추출한 BVP 파형 복원
- 패치 신호 분석
- 탐지용 rPPG 특징 5개 계산

즉, `pyVHR/`가 신호를 추출하고 `features/`가 해당 신호를 영상 탐지에 사용할 수 있는 수치 특징으로 변환합니다.

---



## 원본 D3 결과와 완전히 같지 않은 이유

전처리 규칙은 원본 D3의 `datasets.py`와 비슷하게 구성되어 있습니다.

- 224 크기 조정
- center crop
- ImageNet normalize

그러나 프레임 선택 방식은 다릅니다.

- 원본 `video2frame`
  - 영상의 3초 구간만 사용
  - 8fps로 프레임 저장
- 현재 `visual/frames.py`
  - 영상 전체에서 프레임 선택
  - 8~16장을 균등하게 샘플링
  - 디스크에 jpg를 저장하지 않고 메모리에서 처리

따라서 현재 프로젝트의 결과는 원본 D3 `eval.py`에서 계산한 AP 수치와 완전히 같지 않을 수 있습니다.

원본 논문의 AP 결과를 재현하거나 비교해야 한다면 원본 D3의 다음 흐름을 별도로 사용해야 합니다.

```text
video2frame
 → folder2csv
 → eval.py
```

---



# 데이터 구성

기본 데이터 폴더는 다음과 같습니다.

```text
data/videos/
├── real/
│   └── 실제 영상 
└── fake/
    └── AI 생성·조작 영상 
```

`real`과 `fake` 폴더는 영상 처리 방법을 변경하지 않습니다.

모든 영상은 동일한 파이프라인으로 처리되며, 폴더명은 결과 CSV의 `label` 컬럼을 자동으로 생성하는 데 사용됩니다.

```text
data/videos/real/*.mp4 → label=real
data/videos/fake/*.mp4 → label=fake
```

한 폴더에 영상을 모두 저장한 경우에는 `--real-stems` 옵션으로 실제 영상 파일명을 지정할 수 있습니다.

```bash
python run.py /path/to/videos/ \
  --real-stems subject1,subject11
```

---



# 출력 CSV 컬럼


| 구분         | 컬럼                                                                                    |
| ---------- | ------------------------------------------------------------------------------------- |
| 메타데이터      | `video_name`, `label`, `success`                                                      |
| rPPG       | `absdiff_std`, `bvp_std`, `patch_corr_mean`, `patch_signal_std_mean`, `pseudo_snr_db` |
| D3 및 시각 특징 | `d3_temporal_score`, `highfreq_score`                                                 |
| 오류         | `error`                                                                               |


`error` 컬럼은 하나 이상의 영상 처리에 실패한 경우 생성됩니다.

## 주요 컬럼 설명



### `video_name`

처리한 영상 파일명입니다.

### `label`

영상의 정답 라벨입니다.

```text
real
fake
```



### `success`

영상의 모든 특징 추출이 성공했는지 나타냅니다.

```text
True
False
```



### `error`

특징 추출 실패 시 오류 내용을 저장합니다.

### `absdiff_std`

BVP 신호의 연속 차이값에 대한 표준편차입니다.

### `bvp_std`

복원된 BVP 파형의 표준편차입니다.

### `patch_corr_mean`

얼굴 피부 패치 신호 간 평균 상관관계입니다.

### `patch_signal_std_mean`

피부 패치별 신호 표준편차의 평균입니다.

### `pseudo_snr_db`

생체신호 주파수 대역을 이용해 계산한 의사 SNR 값입니다.

### `d3_temporal_score`

연속 프레임의 시각 특징 변화량을 이용한 시간적 불연속성 점수입니다.

### `highfreq_score`

Laplacian 기반 프레임 고주파 및 질감 특징 점수입니다.

---



# 프로젝트 구조

```text
NOC_AI-Detection/
├── run.py
├── requirements.txt
├── README.md
├── data/
│   └── videos/
│       ├── real/
│       └── fake/
├── pyVHR/
│   ├── analysis/
│   ├── extraction/
│   ├── BVP/
│   └── BPM/
├── visual/
│   ├── d3_model.py
│   ├── frames.py
│   └── highfreq.py
├── features/
│   ├── io.py
│   └── rppg_metrics.py
└── results/
```

---



# 자주 발생하는 문제


| 증상                                              | 원인 및 해결 방법                                            |
| ----------------------------------------------- | ----------------------------------------------------- |
| `No module named 'torch'`                       | 가상환경을 활성화한 후 `pip install -r requirements.txt`를 다시 실행 |
| `No videos found`                               | 입력 경로와 `data/videos/` 내부에 mp4 파일이 있는지 확인              |
| MediaPipe 오류                                    | Python 3.9~3.12 환경인지 확인 후 MediaPipe 재설치               |
| OpenCV 오류                                       | `pip install --upgrade opencv-python` 실행              |
| 얼굴 미검출                                          | 정면 얼굴이 충분히 크게 등장하는 영상인지 확인                            |
| `success=False`                                 | CSV의 `error` 컬럼에서 실패 원인 확인                            |
| CUDA 관련 오류                                      | CUDA 지원 PyTorch와 NVIDIA 드라이버 설치 상태 확인                 |
| `Torch not compiled with CUDA enabled`          | CPU 전용 PyTorch가 설치된 상태이므로 CUDA 지원 빌드로 재설치             |
| `CUDA out of memory`                            | `--workers` 낮추기, 더 가벼운 인코더 사용, 다른 GPU 작업 종료 또는 CPU 실행 |
| 워커를 올려도 느림 / 시스템 응답 저하                         | CPU·RAM 포화일 수 있음. `--workers`를 줄이거나 다른 프로그램을 종료          |
| `Expected all tensors to be on the same device` | 모델과 입력 프레임이 같은 장치로 이동했는지 확인                           |
| 첫 실행이 오래 걸림                                     | 인코더 가중치 다운로드와 초기 모델 로딩 때문일 수 있음                       |


---



## CUDA가 `False`로 표시되는 경우

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

결과:

```text
False
```

다음을 확인합니다.

1. NVIDIA GPU가 장착되어 있는지
2. NVIDIA 드라이버가 설치되어 있는지
3. 현재 가상환경에 CUDA 지원 PyTorch가 설치되어 있는지
4. 터미널이 올바른 Python 가상환경을 사용하고 있는지

현재 Python 경로를 확인합니다.

### macOS / Linux

```bash
which python
```



### Windows

```cmd
where python
```

PyTorch 설치 정보를 확인합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

---



# 성능 비교 방법

CPU와 GPU 실행 시간을 비교하려면 같은 영상과 같은 인코더를 사용해야 합니다.

## macOS / Linux



### CPU

```bash
time python run.py data/videos/fake/subject13_ai.mp4 \
  --device cpu \
  --save-dir results_cpu/
```



### GPU

```bash
time python run.py data/videos/fake/subject13_ai.mp4 \
  --device cuda \
  --save-dir results_gpu/
```



## Windows PowerShell

```powershell
Measure-Command {
    python run.py data/videos/fake/subject13_ai.mp4 --device cpu --save-dir results_cpu/
}
```

```powershell
Measure-Command {
    python run.py data/videos/fake/subject13_ai.mp4 --device cuda --save-dir results_gpu/
}
```

정확한 비교를 위해 다음 조건을 동일하게 유지합니다.

- 같은 영상
- 같은 인코더
- 같은 loss 방식
- 같은 Python 환경
- 같은 프레임 샘플링 설정
- 같은 백그라운드 작업 상태

첫 GPU 실행에는 모델 초기화와 가중치 로딩 시간이 포함될 수 있으므로, 여러 번 실행한 평균 시간을 비교하는 것이 좋습니다.

---



# 현재 한계

- rPPG POS 방식은 CPU에서 실행됩니다. 전체 데이터셋 속도는 `--workers` 병렬화가 더 효과적인 경우가 많습니다.
- `--device auto`는 CUDA 사용 가능 시 GPU를 선택하고, 없으면 CPU를 사용합니다. `--device cuda`는 CUDA가 없으면 오류를 냅니다.
- GPU 가속 성능은 D3 인코더와 영상 조건에 따라 달라집니다.
- `--workers`를 올리면 워커마다 모델을 로드하므로 RAM·VRAM 사용량이 증가합니다.
- 이 프로젝트의 결과 CSV 자체는 최종 real/fake 판정 결과가 아니라 탐지용 특징값입니다.
- 최종 분류를 위해서는 추출 특징을 이용한 별도의 분류 모델 또는 판정 기준이 필요합니다.
- 현재 프레임 샘플링 방식은 원본 D3 논문의 평가 방식과 다릅니다.
- 원본 D3의 AP 성능을 그대로 재현한다고 볼 수 없습니다.

---



# 원본 프로젝트와의 관계

- 기존 `D3/`와 `pyVHR/` 프로젝트에서 실제 사용하는 코드만 가져와 통합했습니다.
- 원본 프로젝트는 백업 및 재현 목적으로 별도로 유지할 수 있습니다.
- 이후 통합 특징 추출 작업은 이 저장소의 `run.py`를 중심으로 진행합니다.
- 원본 D3 평가 결과를 재현해야 하는 경우 원본 전처리 및 `eval.py` 흐름을 별도로 사용해야 합니다.

---



# 빠른 실행 요약



## CPU

```bash
git clone https://github.com/dokyung17/NOC_AI-Detection.git
cd NOC_AI-Detection

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python run.py data/videos/real/subject1.mp4 --device cpu
```



## NVIDIA GPU

```bash
git clone https://github.com/dokyung17/NOC_AI-Detection.git
cd NOC_AI-Detection

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"

python run.py data/videos/real/subject1.mp4 --device cuda
```

전체 데이터셋을 GPU로 실행하려면 다음 명령을 사용합니다.

```bash
python run.py --device cuda
```

영상 개수가 많을 때는 병렬 워커를 함께 사용합니다.

```bash
python run.py --device cuda --workers 4
```

