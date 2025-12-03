# 📘 Colored MNIST Classification
**Soongsil University · Machine Learning (2025-2)**  
**Team ML-SSU-2025**

전통적인 머신러닝(Classical ML) 기법만을 사용하여 **Colored MNIST** 데이터셋에 대해 다음 세 가지 분류 과제를 수행하는 프로젝트입니다.

1. **Digit Classification** (0–9 숫자 분류)
2. **Foreground Color Classification** (전경색 7-class 분류)
3. **Background Color Classification** (배경색 7-class 분류)

> 🚫 본 프로젝트에서는 CNN, MLP 등 인공신경망 기반 모델을 사용하지 않습니다.  
> ✅ KNN, SVM, Decision Tree, Random Forest, XGBoost 중심으로 진행합니다.

---

## 🧭 Project Overview

### 목표

- MNIST 이미지를 기반으로 **전경색(FG) / 배경색(BG)** 이 다른 컬러 이미지 데이터셋(Colored MNIST)을 생성
- 색상 간 상관관계, 숫자·색 정보 분리 가능성 등을 **전통적 ML 모델**로 분석
- 각 Task에 대해:
  - **Accuracy / Precision / Recall / F1-score**
  - **Confusion Matrix**
  - **(Tree 계열) Feature Importance 시각화**
- 재현 가능한 파이프라인 및 협업 환경 구축 (로컬 환경 중심)

### 3가지 과제 정의

| Task | Input | Output Classes | 설명 |
|------|------|----------------|------|
| Digit | Colored digit image | 10 (0–9) | 숫자 인식 |
| Foreground Color | Colored digit image | 7 (R,O,Y,G,B,I,V) | 전경(숫자) 색상 분류 |
| Background Color | Colored digit image | 7 (R,O,Y,G,B,I,V) | 배경 색상 분류 |

색상 팔레트는 RAINBOW(ROYG BIV) 7색을 사용합니다.

---

## ⚙️ Environment

### Recommended

- Python `3.9 ~ 3.11`
- OS: Windows / macOS / Linux
- IDE: VS Code, Jupyter Notebook / Lab

### Required Packages

`requirements.txt` 예시 (또는 이와 동등한 환경):

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

실제 프로젝트에서는 `pip freeze > requirements.txt` 또는 `requirements.lock.txt` 등을 통해 최종 제출 시점의 환경을 고정하는 것을 권장합니다.

⸻

## 🧱 Repository Structure

```
colored-mnist-classification/
│
├── notebooks/
│   ├── 01_preprocessing_colored_mnist.ipynb   # Colored MNIST 생성 + 전처리 + EDA
│   ├── 02_train_classical_ml.ipynb            # 3개 Task × 5개 모델(val/test 평가, GridSearch 일부)
│   └── 03_evaluation_and_plots.ipynb          # 리포트용 재평가(혼동행렬/오분류/정규화 결과)
│
├── data/
│   ├── raw/
│   │   ├── mnist/
│   │   │   └── mnist_train.npz                # 원본 MNIST (로컬 전용, Git 미추적)
│   │   ├── fonts/                             # 폰트 파일 (선택, 로컬 전용)
│   │   └── ml_project_original/               # 기타 원본 백업
│   └── processed/
│       └── colored_mnist/
│           └── colored_mnist.npz              # 전처리 결과 (로컬 전용, Git 미추적)
│
├── results/
│   ├── metrics/                               # CSV 결과 (로컬 전용)
│   └── figures/                               # 시각화 이미지 (로컬 전용)
│
├── docs/
│   ├── meeting_notes/
│   ├── reports/
│   └── slides/
│
├── .gitignore
├── requirements.txt
└── README.md
```

### `.gitignore` 핵심 규칙

데이터 & 대용량 산출물은 GitHub에 올리지 않습니다.

```
data/
results/
notebooks/data/

*.npz
*.npy
*.pkl
*.pt
*.pth
*.onnx
*.log

.ipynb_checkpoints/
__pycache__/
```

⸻

## 🚀 Local Setup Guide

### 1️⃣ Clone & Branch

```bash
git clone https://github.com/ML-SSU-2025/colored-mnist-classification.git
cd colored-mnist-classification

# 작업용 브랜치
git checkout develop
git pull origin develop
```

`main` 은 최종 제출용/보호 브랜치, 실 작업은 항상 `develop` 또는 `nb/<topic>` 브랜치에서 진행합니다.

⸻

### 2️⃣ Python 환경 설정

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

# 공통
pip install --upgrade pip
pip install -r requirements.txt  # 혹은 직접 필요한 패키지 설치
```

⸻

### 3️⃣ 데이터 배치

프로젝트 루트에서 다음 구조를 만족하도록 파일을 준비합니다.

```bash
mkdir -p data/raw/mnist
mkdir -p data/raw/fonts    # 선택
```

- `data/raw/mnist/mnist_train.npz`
  - `train_images (N, 28, 28)`, `train_labels (N,)` 구조의 NPZ
  - Git에 포함하지 않음 (직접 복사)
- `data/raw/fonts/*.ttf` (선택)
  - 예: MaruBuri 계열 폰트
  - 폰트 기반 synthetic digits 생성 시 사용

---

## 🧪 01_preprocessing_colored_mnist.ipynb

**주요 기능**
1. **MNIST 로드**
   - `mnist_train.npz`에서 `train_images`, `train_labels` 로드
   - [0, 1] 스케일로 정규화
2. **Colored MNIST 생성**
   - 7개 색상 팔레트 (RED, ORANGE, YELLOW, GREEN, BLUE, INDIGO, VIOLET)
   - 전경/배경 색상 인덱스를 무작위 선택 (fg ≠ bg)
   - threshold 기반으로 digit 픽셀과 background 분리 후 RGB 매핑
   - 결과를 `(28*28*3,)` 벡터로 평탄화
3. **폰트 기반 샘플 (선택)**
   - `data/raw/fonts/*.ttf` 존재 시 TTF 폰트로 0–9 숫자 렌더링
   - bounding box 기반 center-align 후 동일한 색상 변환 적용
4. **데이터 증강 (Augmentation)**
   - Grayscale 단계에서 작은 회전(±15°), 평행이동(±2px), 약한 Gaussian 노이즈 적용
5. **EDA**
   - Original vs Colored 시각화, Font-based 샘플, 라벨 분포, Digit×Color 교차분포 확인
6. **최종 Merge & Split**
   - base/aug/font 샘플 통합 후 셔플, **Train/Val/Test = 0.64 / 0.16 / 0.20** (test_size=0.2, train 내부에서 val 20%)
   - 모든 split이 `stratify=y_digit` 조건을 만족하도록 분할
   - `X_train` 기준 feature-wise mean/std 계산 후 동일 통계를 val/test에 적용

**출력**: `data/processed/colored_mnist/colored_mnist.npz`
- `X_train`, `X_val`, `X_test` (표준화; mean/std는 train에서 계산)
- `X_train_raw`, `X_val_raw`, `X_test_raw`
- `y_digit_*`, `y_fg_*`, `y_bg_*` (train/val/test)
- `source_train`, `source_val`, `source_test` (데이터 출처 태그)
- `mean`, `std`

이 파일은 로컬 전용이며 Git에 커밋하지 않습니다.

---

## 🤖 02_train_classical_ml.ipynb

`data/processed/colored_mnist/colored_mnist.npz`(train/val/test, 표준화된 features + raw features)을 로드하여 **3개 Task × 5개 모델**을 학습/평가하고 리포트용 지표를 정리합니다. 노트북 상단의 `TASK` 값을 `digit` / `fg` / `bg` 중 하나로 선택해 라벨을 바꿉니다.

### 대상 Task
- `digit`: 10-way classification
- `fg` (foreground color): 7-way classification (ROYGBIV)
- `bg` (background color): 7-way classification (ROYGBIV)

### 사용 모델
1. `KNeighborsClassifier` + GridSearchCV(k, weights, p)
2. `SVC (RBF)` + GridSearchCV(C, gamma)
3. `DecisionTreeClassifier` (baseline)
4. `RandomForestClassifier` (500 trees, entropy)
5. `xgboost.XGBClassifier` (early stopping, hist)

모든 모델은 `random_state=42`로 고정하며 표준화된 입력을 사용합니다.

### 평가/동작
- 공통 함수 `evaluate_classifier`가 train 후 **val/test** 모두에 대해 Accuracy, Precision/Recall/F1(weighted)과 Confusion Matrix를 출력합니다.
- KNN/SVM은 작은 그리드 서치, XGBoost는 early stopping(`eval_set=[(X_val, y_val)]`)을 사용합니다.
- 현재는 결과물(CSV/PNG)을 자동 저장하지 않고, 노트북에서 바로 지표와 플롯을 확인하는 형태입니다(리포트용으로 필요 시 수동 저장).

### 실행 흐름
1. `colored_mnist.npz`가 존재하는지 확인 (`data/processed/colored_mnist/colored_mnist.npz`).
2. `TASK`를 원하는 과제로 설정.
3. XGBoost가 없다면 상단 셀에서 설치 후 런타임 재시작(Colab 등).
4. 각 모델 셀을 순차 실행 → 콘솔 로그와 플롯으로 결과 확인.

### 출력
- 콘솔 로그: val/test 지표 및 best hyperparameters
- 플롯: val/test 혼동행렬 (즉시 표시)
- 필요 시 지표/그림 저장 코드는 추후 추가 예정

---

## 📊 03_evaluation_and_plots.ipynb (리포트 정리)

- 입력: 01에서 생성한 `colored_mnist.npz`를 로드.
- `TARGET_MODEL_NAME`를 02에서 가장 성능이 좋았던 모델 이름으로 설정(예: `"rf"`, `"xgb"`, `"svm"` 등).
- 각 task (`digit`, `fg`, `bg`)에 대해 동일한 설정으로 **다시 학습(train→val)** 후 리포트용 결과를 생성:
  - Confusion matrix 2종: raw count / normalized 비율.
  - Class-wise accuracy 테이블 + bar plot.
  - Source-wise accuracy 테이블 + bar plot(데이터 출처별 성능 비교).
  - 오분류 샘플 CSV 저장 + 이미지 시각화.
  - 가장 많이 헷갈리는 `(true → pred)` 조합 상위 몇 건을 출력.
- 발표/보고서에 바로 사용할 수 있는 표와 그림을 한 번에 모으는 용도입니다.

---

## 🧩 Collaboration Guide

### Branch Strategy
- `main`: 최종 결과/제출용 (직접 push 금지)
- `develop`: 공용 개발 브랜치, 모든 기능 분기의 기준
- `nb/<topic>`: 개인 또는 기능 브랜치 (예: `nb/preprocessing-fix`, `nb/knn-tuning`)

### Commit Rules
- `data/`, `results/`, `*.npz`, `*.png` 등 대용량 산출물은 커밋 금지
- 변경 대상: `notebooks/*.ipynb`, `README.md`, `docs/`, 설정 파일 등

커밋 메시지 예시
- `[nb] update preprocessing pipeline`
- `[nb] add classical ML training notebook`
- `[conf] update .gitignore for data/results`

### Typical Workflow
1. 최신 develop 가져오기
   ```bash
   git checkout develop
   git pull origin develop
   ```
2. 작업 브랜치 생성
   ```bash
   git checkout -b nb/<topic>
   ```
3. 노트북 수정/실행/검증
4. 변경된 노트북만 커밋
   ```bash
   git add notebooks/01_preprocessing_colored_mnist.ipynb
   git commit -m "[nb] refine preprocessing"
   git push -u origin nb/<topic>
   ```
5. GitHub에서 PR 생성 (base: develop, compare: nb/<topic>)
6. 리뷰 후 merge → 작업 브랜치 삭제

---

## 👥 Team & Roles

| Role | Name | Responsibility |
|------|------|----------------|
| Team Lead | 정재훈 | 전체 총괄, 일정 관리, GitHub 구조 설계 |
| Data | 정재훈 | Colored MNIST 설계 및 전처리 파이프라인 |
| Data | 최은지 | 데이터 전처리, 색상/폰트 샘플링, EDA |
| Modeling | 정재훈 | XGBoost 튜닝 및 평가 분석 |
| Modeling | 이재민 | KNN 튜닝 및 평가 분석 |
| Modeling | 성도연 | SVM 튜닝 및 평가 분석 |
| Evaluation | 이다정 | Decision Tree / Random Forest 튜닝 및 평가 분석 |

---

## 🧾 Report Checklist

최종 보고서 및 발표 자료에 반드시 포함할 항목
1. **데이터 생성 방식**
   - MNIST → Colored MNIST 변환 로직
   - 색상 선택 규칙, threshold, augmentation 전략
2. **모델 선정 근거**
   - 전통적 ML 모델 선택 이유
3. **평가 지표**
   - Accuracy/Precision/Recall/F1 의미 및 해석
4. **결과 분석**
   - Task별 best model, 오분류 패턴 분석
5. **Feature Importance**
   - RandomForest 기반 픽셀 중요도 해석
6. **한계 및 향후 개선**
   - threshold 기반 분리의 한계, color/shape bias, domain generalization 이슈
   - (선택) Neural Network 대비 가능성 언급(구현 X)

---

## 🔍 Reference
- Original MNIST dataset: http://yann.lecun.com/exdb/mnist/
- Colored MNIST / CMNIST 개념: Invariant Risk Minimization, Domain Generalization 연구 계열

---

## 💡 Maintainer

문의 / 이슈 / PR 환영합니다.
- Maintainer: 정재훈 (Jaehoon Jung)
- GitHub: ML-SSU-2025 org 내 운영

---

## 📌 현재 진행 상황 & 향후 발전 방향

### 진행 상황
- **데이터 파이프라인**: 01 노트북으로 Colored MNIST를 생성/표준화하여 `data/processed/colored_mnist/colored_mnist.npz`에 train/val/test 및 raw/표준화된 features를 모두 저장.
- **모델링 노트북(02)**: 단일 `TASK` 선택 방식으로 digit/fg/bg 중 하나를 평가. KNN·SVM은 간단한 GridSearchCV, XGBoost는 early stopping 적용. 공통 평가 함수로 val/test 지표와 혼동행렬을 즉시 확인 가능(리포트용 비교).
- **리포트 노트북(03)**: 02에서 선정한 `TARGET_MODEL_NAME`으로 digit/fg/bg를 각각 재학습(train→val)하여 confusion matrix(카운트/정규화), class/source-wise accuracy 테이블·bar plot, 오분류 CSV·이미지, 상위 혼동 조합을 산출.
- **지표 저장**: 02는 화면 출력 중심, 03은 오분류 CSV 등 일부 산출물을 저장하며 플롯은 노트북에서 바로 확인.

### 향후 발전 방향
1. **결과 저장 자동화**: 02/03 모두에서 지표·플롯을 지정 경로(`results/metrics`, `results/figures`)에 자동 저장하는 옵션 추가.
2. **다중 Task 배치 실행**: digit/fg/bg를 한 번에 돌려 공통 포맷으로 결과를 묶어주는 루프/CLI 스크립트 추가.
3. **하이퍼파라미터 관리**: GridSearch 공간/seed를 config로 분리하고, 실행 시간을 고려한 프리셋(빠른/정밀) 제공.
4. **실행 로그/재현성**: 실행 시점, 사용 모델, best params를 요약해 남기는 로거/요약 셀 추가.
5. **리포트 자동화**: 03 노트북에서 생성된 표/이미지·오분류 CSV를 발표/문서용 폴더에 일괄 내보내는 스크립트화.
