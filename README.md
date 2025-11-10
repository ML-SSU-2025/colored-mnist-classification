# 📘 Colored MNIST Classification
**Soongsil University · Machine Learning (2025-2)**  
**Team ML-SSU-2025**

전통적인 머신러닝(Classical ML) 기법만을 사용하여 **Colored MNIST** 데이터셋에 대해 다음 세 가지 분류 과제를 수행하는 프로젝트입니다.

1. **Digit Classification** (0–9 숫자 분류)
2. **Foreground Color Classification** (전경색 7-class 분류)
3. **Background Color Classification** (배경색 7-class 분류)

> 🚫 본 프로젝트에서는 CNN, MLP 등 인공신경망 기반 모델을 사용하지 않습니다.  
> ✅ KNN, SVM, Decision Tree, Random Forest, Logistic Regression 중심으로 진행합니다.

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
│   ├── 02_train_classical_ml.ipynb            # 3개 Task × 5개 모델 학습 및 평가
│   └── 03_evaluation_and_plots.ipynb          # 결과 해석, 지표 정리, 그림 생성
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
   - base/aug/font 샘플 통합 후 셔플, `train/test split`(test_size=0.2, stratify=y_digit)
   - `X_train` 기준 feature-wise mean/std 계산 후 표준화 버전 저장

**출력**: `data/processed/colored_mnist/colored_mnist.npz`
- `X_train`, `X_test` (표준화)
- `X_train_raw`, `X_test_raw`
- `y_digit_*`, `y_fg_*`, `y_bg_*`
- `mean`, `std`

이 파일은 로컬 전용이며 Git에 커밋하지 않습니다.

---

## 🤖 02_train_classical_ml.ipynb

`colored_mnist.npz`를 로드하여 **3개 Task × 5개 모델**을 학습/평가합니다.

### 대상 Task
- `digit`: 10-way classification
- `fg_color`: 7-way classification
- `bg_color`: 7-way classification

### 사용 모델
1. `KNeighborsClassifier`
2. `SVC (RBF kernel)`
3. `DecisionTreeClassifier`
4. `RandomForestClassifier`
5. `LogisticRegression (multinomial)`

모든 모델은 `random_state`를 고정하고 표준화된 입력을 사용합니다.

### 평가 지표
각 (Task, Model) 조합에 대해
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion Matrix (시각화)

### 출력
- `results/metrics/classical_ml_summary.csv`
- Task별 세부: `results/metrics/digit_model_performance.csv`, `results/metrics/fg_color_model_performance.csv`, `results/metrics/bg_color_model_performance.csv`
- Confusion Matrix 이미지: `results/figures/cm_{task}_{model}.png`
- RandomForest Feature Importance heatmap: `results/figures/fi_{task}_rf.png`

모든 결과물은 보고서/PPT에 바로 활용 가능한 형식을 목표로 합니다.

---

## 📊 03_evaluation_and_plots.ipynb (선택)

- 모델 결과 요약표 정리
- best 모델 하이라이트, 오분류 사례 캡처
- 발표용 플롯/테이블 생성
- `results/figures/` 및 `docs/` 자료와 연동

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
| Data | 최은지 | 색상/폰트 샘플링, EDA 보조 |
| Modeling | 이재민 | Decision Tree / Random Forest |
| Modeling | 성도연 | KNN / SVM 튜닝 및 비교 |
| Evaluation | 이다정 | Confusion Matrix, 지표 정리, 표 생성 |

(역할 분담은 프로젝트 진행 상황에 따라 조정 가능합니다.)

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


