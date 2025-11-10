# Colored MNIST 고전 ML 파이프라인

이 저장소는 Colored MNIST 데이터를 전통적인 머신러닝 기법으로 다루기 위한 표준 구조를 제공합니다. 모든 실험은 주피터 노트북 기반으로 진행하며, 팀 전원이 같은 파이프라인을 재현할 수 있도록 구성했습니다.

## 범위 및 규칙
- 허용 모델: **KNN, SVM, 의사결정나무, 랜덤포레스트, 로지스틱 회귀**
- 예측 대상: **숫자 레이블**, **전경(글자) 색상**, **배경 색상**의 세 가지
- MLP, CNN 등 **신경망 모델은 금지**입니다. 관련 코드·문서·폴더를 추가하지 마세요.

## 공식 노트북 파이프라인
| 단계 | 노트북 | 설명 |
| --- | --- | --- |
| 전처리 | `notebooks/01_preprocessing_colored_mnist.ipynb` | `data/raw/mnist/mnist_train.npz`와 MaruBuri 폰트를 불러와 `data/processed/colored_mnist/colored_mnist.npz` 생성 |
| 학습 | `notebooks/02_train_classical_ml.ipynb` | 세 과제(숫자, 전경색, 배경색)에 대해 KNN/SVM/DT/RF/LogReg 학습 |
| 평가 | `notebooks/03_evaluation_and_plots.ipynb` | 혼동행렬, 지표 표, 발표용 그림 등 산출 |

위 세 노트북만이 **공식 파이프라인**입니다. 재현 시 반드시 순서대로 실행하세요.

## 노트북 디렉터리
```
notebooks/
├── 01_preprocessing_colored_mnist.ipynb   # 공식 전처리
├── 02_train_classical_ml.ipynb            # 공식 학습
├── 03_evaluation_and_plots.ipynb          # 공식 평가
├── team_nb_jaehoon.ipynb                  # 개인 실험 노트
├── team_nb_template.ipynb                 # 팀원용 템플릿
└── legacy/
    ├── gen_mnist_original.ipynb           # ML_Project.zip 원본
    └── README.md                          # 레거시 정책 안내
```
- 새 실험은 템플릿을 복제해 사용하고, 반드시 클래식 ML 모델만 활용합니다.
- `legacy/` 폴더는 과거 자료 보관용이며 공식 파이프라인에는 포함되지 않습니다.

## 데이터 구조 및 보관 정책
```
data/
├── raw/
│   ├── mnist/mnist_train.npz             # ML_Project.zip에서 복사
│   ├── fonts/MaruBuri-*.ttf              # ML_Project.zip 폰트 일괄 보관
│   └── ml_project_original/              # 기타 원본 파일 백업
└── processed/
    └── colored_mnist/colored_mnist.npz   # 전처리 노트북 출력
```
- `data/`와 `results/`는 전부 `.gitignore` 처리되어 Git에 올라가지 않습니다.
- 새로운 원본 데이터가 생기면 수정 없이 `data/raw/ml_project_original/` 안에 보관하세요.

## 결과 및 문서화
```
results/
├── metrics/   # 평가 지표(CSV/JSON 등)
└── figures/   # 시각화/그래프/혼동행렬
```
- 결과물은 로컬 또는 Drive에만 저장하고 Git에는 커밋하지 않습니다.
- 문서 자료는 `docs/`에 정리합니다.
```
docs/
├── meeting_notes/
├── reports/
│   ├── decisions.md
│   └── references.md
└── slides/
```

## 시작 방법
1. `pip install -r requirements.txt` (또는 `environment.yml`로 Conda/Mamba 세팅)
2. `ML_Project.zip`에서 받은 `mnist_train.npz`와 MaruBuri 폰트를 `data/raw/` 구조에 맞춰 배치
3. 노트북을 전처리 → 학습 → 평가 순서로 실행
4. 산출물은 `results/`에 저장하고 주요 내용은 `docs/`에 기록

이 구조를 따르면 과거 자료는 안전하게 유지되고, 클래식 ML 기반 워크플로우를 간단히 재현할 수 있습니다.
