# Colored MNIST 전통 머신러닝 분류 프로젝트

**팀명: 전과자**  
**과목: 머신러닝 (2025-2)**  
**소속: 숭실대학교**

---

## 프로젝트 개요

본 프로젝트는 원본 MNIST에 전경(Foreground) 및 배경(Background) 색상을 7종 RGB 팔레트로 적용해 구성한 **Colored MNIST**를 대상으로, 전통적 머신러닝(Classical ML) 모델들의 분류 성능을 체계적으로 비교·분석합니다.

### 세 가지 분류 과제

| Task | Input | Output Classes | 설명 |
|------|-------|----------------|------|
| **Digit** | Colored digit image | 10 (0-9) | 숫자 인식 |
| **Foreground Color** | Colored digit image | 7 (ROYGBIV) | 전경(숫자) 색상 분류 |
| **Background Color** | Colored digit image | 7 (ROYGBIV) | 배경 색상 분류 |

### 주요 성능 결과

| 모델 | Digit | FG | BG | 특징 |
|------|-------|-----|-----|------|
| **KNN (PCA)** | **97.93%** | - | - | PCA 적용 시 최고 성능 (별도 실험) |
| **XGBoost** | **95.27%** | 99.99% | 100% | 종합 최고 성능 |
| **SVM** | 94.24% | 99.74% | 100% | 비선형 경계 학습 |
| **Random Forest** | 93.46% | 100% | 100% | 안정적 성능 |
| **Decision Tree** | 77.13% | 99.31% | 100% | 해석 가능 |
| **KNN (원본)** | 91.57% | 91.57% | 99.99% | 거리 기반 분류 |

---

## 프로젝트 구조

```
colored-mnist-classification/
├── 프로젝트_소스코드_전과자.ipynb  # 통합 노트북 (Colab 실행용)
├── README.md                        # 본 파일
├── requirements.txt                 # 필수 패키지 목록
├── configs/                         # 설정 파일
│   ├── models.yaml                  # 모델 하이퍼파라미터
│   ├── paths.yaml                   # 경로 설정
│   └── experiments/                 # Task별 실험 설정
├── docs/
│   └── reports/
│       └── final_report.md          # 최종 보고서
└── results/                         # 실험 결과
    ├── metrics/                     # 성능 지표 (CSV)
    ├── figures/                     # 시각화 결과 (PNG)
    ├── error_analysis/              # 오분류 분석
    └── models/                      # 학습된 모델 (joblib)
```

---

## 실행 방법

### Google Colab에서 실행

1. **노트북 업로드**
   - `프로젝트_소스코드_전과자.ipynb`를 Google Colab에 업로드

2. **데이터 파일 준비**
   - `mnist_train.npz` 파일을 Google Drive에 업로드
   - 경로: `/content/drive/MyDrive/colored-mnist-classification/data/raw/mnist/mnist_train.npz`
   - 폰트 파일(선택): `/content/drive/MyDrive/colored-mnist-classification/data/raw/fonts/`

3. **Config 파일 준비**
   - `configs/` 디렉토리를 Google Drive에 업로드
   - 경로: `/content/drive/MyDrive/colored-mnist-classification/configs/`

4. **실행**
   - 노트북의 셀을 순서대로 실행
   - 예상 실행 시간: 약 2-3시간 (모델 학습 시간 포함)

### 로컬 환경에서 실행

1. **환경 설정**
   ```bash
   pip install -r requirements.txt
   ```

2. **데이터 준비**
   - `data/raw/mnist/mnist_train.npz` 파일 준비
   - 폰트 파일(선택): `data/raw/fonts/*.ttf`

3. **실행**
   - Jupyter Notebook 또는 JupyterLab에서 `프로젝트_소스코드_전과자.ipynb` 실행

---

## 테스트 셋 평가 방법 (교수님용)

이미 학습된 모델을 사용하여 테스트 셋만 평가하는 방법입니다.

### 필요한 파일

1. **학습된 모델 파일**
   - `results/models/` 디렉토리에 모든 모델 파일(.joblib)이 있어야 함
   - 예: `digit_knn.joblib`, `digit_svm.joblib`, `fg_rf.joblib` 등

2. **테스트 셋 파일**
   - 교수님이 제공한 테스트 셋 파일 (`.npz` 형식)
   - 예상 구조: `X_test`, `y_digit_test`, `y_fg_test`, `y_bg_test`

3. **Config 파일**
   - `configs/paths.yaml`: 테스트 셋 경로 설정
   - `configs/models.yaml`: 모델 설정 (참고용)

### 실행 방법

1. **테스트 셋 경로 설정**
   ```yaml
   # configs/paths.yaml
   paths:
     data:
       test: "data/test/professor_test_set.npz"  # 테스트 셋 경로
   ```

2. **노트북 실행**
   - `테스트_셋_평가_전용.ipynb`를 Jupyter/Colab에서 열기
   - 셀을 순서대로 실행

3. **결과 확인**
   - `results/metrics/test_metrics_all_models.csv`: 전체 모델 성능 요약
   - `results/figures/cm_*_test_*.png`: Confusion Matrix 시각화

### 주의사항

- 학습된 모델 파일이 없으면 평가가 불가능합니다
- 테스트 셋은 학습 시 사용한 전처리(StandardScaler, PCA 등)가 자동으로 적용됩니다
- 모델 파일은 Pipeline 형태로 저장되어 있어 별도 전처리 코드가 필요 없습니다

---

## 주요 특징

### 1. 체계적인 하이퍼파라미터 튜닝
- 단순 GridSearch를 넘어서 각 파라미터의 물리적 의미를 이해하고 실험 설계
- Task별 최적 하이퍼파라미터 도출 (예: KNN의 형태 분류 vs 색상 분류)

### 2. 전처리 기법 최적화
- **Deskewing**: 기울기 보정으로 클래스 내 분산 감소
- **PCA**: KNN에 적용하여 Digit Task에서 97.93% 달성
- **StandardScaler**: 거리 기반 모델의 성능 향상

### 3. 데이터 증강
- **Font-based**: 외부 폰트로 형태적 다양성 확보
- **Geometric**: Shift/Rotation으로 강건성 향상
- 최적 비율 탐색을 통한 효과적 증강

### 4. 모델별 특성 분석
- KNN: 거리 기반 분류, PCA 적용 시 최고 성능
- SVM: 비선형 경계 학습, 하이퍼파라미터 튜닝 중요
- Decision Tree: 해석 가능, 과적합 위험
- Random Forest: 안정적, variance 감소
- XGBoost: Boosting 효과, Digit Task 최고 성능

---

## 실험 결과 요약

### Digit Classification
- **최고 성능**: KNN + PCA 97.93% (별도 실험)
- **PCA 없이 최고**: XGBoost 95.27%
- 형태적 복잡성으로 모델 간 성능 차이 큼

### Foreground/Background Color Classification
- 대부분의 모델에서 99-100% 정확도
- 색상 정보가 RGB 공간에서 명확히 분리되어 분류 용이

### 주요 발견사항
1. **전처리의 중요성**: KNN에 PCA 적용 시 5%+ 성능 향상
2. **Task별 하이퍼파라미터 최적화**: 형태 분류에는 Manhattan 거리, 색상 분류에는 Euclidean 거리
3. **하이퍼파라미터 튜닝의 효과**: SVM의 경우 C와 gamma 튜닝만으로 7.2% 향상 (87.01% → 94.24%)

---

## 필수 패키지

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
pyyaml
pillow
opencv-python-headless
jupyter
```

자세한 버전 정보는 `requirements.txt`를 참고하세요.

---

## 보고서

최종 보고서는 `docs/reports/final_report.md`에 포함되어 있습니다.

주요 내용:
- 데이터 전처리 및 선택 이유
- 데이터 증강 기법 및 실험 결과
- 모델 선정 근거 및 하이퍼파라미터 튜닝 과정
- 성능 평가 방법 및 결과 분석
- 결론 및 향후 연구 방향

---

## 팀원

정재훈 · 성도연 · 이다정 · 이재민 · 최은지

---

## 참고사항

- 본 프로젝트는 전통적 머신러닝 모델만 사용합니다 (CNN, MLP 등 신경망 제외)
- 모든 실험은 재현 가능하도록 random seed를 고정했습니다
- 상세한 실험 과정과 결과는 최종 보고서를 참고하세요

---

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.
