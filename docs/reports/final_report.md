# Colored MNIST에서의 전통 머신러닝 기반 분류 성능 비교 연구

## (A Comparative Study of Classical Machine Learning Models on the Colored MNIST Dataset)

**팀명: 전과자**

정재훈 · 성도연 · 이다정 · 이재민 · 최은지

---

## 요약 (Abstract)

본 연구는 원본 MNIST에 전경(Foreground) 및 배경(Background) 색상을 7종 RGB 팔레트로 적용해 구성한 Colored MNIST를 대상으로, 전통적 머신러닝(Classical ML) 모델들의 분류 성능을 체계적으로 비교·분석하였다. 실험 과제는 Digit(10 classes), Foreground Color(7 classes), Background Color(7 classes)의 세 가지로 구성되며, 100,000장 규모의 데이터셋을 Train 80% / Validation 20%로 분할하여 학습하였다.

본 연구에서는 KNN, SVM(RBF Kernel), Decision Tree, Random Forest, XGBoost 총 5개 모델을 평가하였으며, 각 모델에 대해 **체계적인 하이퍼파라미터 튜닝 과정**을 거쳐 최적의 성능을 달성하고자 노력하였다. 특히 단순히 GridSearchCV를 실행하는 것을 넘어서, 각 파라미터가 모델 성능에 미치는 영향을 이해하고, 실험 결과를 분석하여 최적값을 도출하는 과정을 상세히 기록하였다.

실험 결과, 색상 기반 분류(FG/BG)는 모든 모델에서 99~100%의 매우 높은 정확도를 기록하여 색상 feature의 분리 난이도가 낮음을 확인하였다. 반면 Digit 분류는 구조적 특징 학습이 필요해 모델 간 차이가 뚜렷하게 나타났으며, **KNN+PCA(97.93%)**가 별도 실험에서 최고 성능을 달성했고, **XGBoost(95.27%)**, **SVM(94.24%)**, **Random Forest(93.46%)**이 우수한 성능을 보였다. 특히 **KNN에 PCA를 적용한 경우 Digit Task에서 97.93%의 최고 성능**을 달성하여, 전처리 기법의 중요성을 실험적으로 입증하였다. 또한 **Task별 하이퍼파라미터 최적화**를 통해 KNN의 경우 형태 분류에는 Manhattan 거리, 색상 분류에는 Euclidean 거리가 더 효과적임을 확인하였다.

본 연구는 색상·형태가 결합된 환경에서 전통적 머신러닝 모델이 갖는 특성과 한계를 명확히 제시하며, **하이퍼파라미터 튜닝, 전처리 최적화, 앙상블 방법 등 다양한 시도를 통해 성능을 향상시키는 과정**을 상세히 기록하여 머신러닝 학습의 실질적인 경험을 공유한다.

**키워드**: Colored MNIST, 머신러닝 분류, 색상 인식, 전통적 ML 모델, XGBoost, Random Forest, PCA, 하이퍼파라미터 튜닝, 성능 최적화

---

## 1. 서론

### 1.1 연구 배경 및 목적

본 프로젝트는 전통적 머신러닝 기법을 활용해 Colored MNIST 데이터셋의 분류 성능을 분석하는 것을 목표로 한다. 기존 MNIST는 흑백 기반의 단순 손글씨 숫자 데이터셋으로, 실제 시각 환경에서 발생하는 다양성(색상, 배경, 조명 변화)을 반영하지 못한다는 한계가 있다.

이에 본 연구에서는 원본 MNIST를 기반으로 숫자의 전경색(Foreground Color)과 배경색(Background Color)을 7종의 RGB 팔레트(ROYGBIV)로 변환한 Colored MNIST를 구축하고, 이 데이터가 전통 ML 모델에서 어떻게 해석되고 분류되는지 체계적으로 분석한다.

**더 나아가, 본 연구는 단순한 성능 비교를 넘어서 "어떻게 하면 성능을 향상시킬 수 있는가?"라는 질문에 대한 실질적인 탐구 과정을 기록한다.** 하이퍼파라미터 튜닝, 전처리 기법 적용, 앙상블 방법 등 다양한 시도를 통해 머신러닝 모델의 성능 최적화 과정을 직접 경험하고, 그 결과를 공유하는 것이 본 연구의 핵심 목표이다.

### 1.2 연구 과제

본 연구는 다음 세 가지 분류 과제를 다룬다:

| Task | Input | Output Classes | 설명 |
|------|-------|----------------|------|
| Digit | Colored digit image | 10 (0-9) | 숫자 인식 |
| Foreground Color | Colored digit image | 7 (ROYGBIV) | 전경(숫자) 색상 분류 |
| Background Color | Colored digit image | 7 (ROYGBIV) | 배경 색상 분류 |

이를 통해 모델이 형태 정보(Shape)와 색상 정보(Color)를 얼마나 잘 구분하는지를 각각 독립적으로 평가할 수 있다.

### 1.3 연구 파이프라인

본 연구의 전체 처리 과정은 다음과 같이 구성된다:

```
MNIST 원본 데이터 (60,000장)
    ↓
전처리 (Deskewing, 정규화)
    ↓
데이터 증강 (Font Synthesis: 10,000장, Geometric: 10,000장)
    ↓
Colorization (FG/BG 색상 적용)
    ↓
Dataset Split (Train 80,000 / Val 20,000)
    ↓
모델 학습 (KNN, SVM, Decision Tree, RF, XGBoost)
    ↓
하이퍼파라미터 튜닝 (GridSearchCV + 수동 실험)
    ↓
앙상블 (Weighted Voting, Stacking)
    ↓
성능 평가 및 분석
```

---

## 2. 데이터 전처리

### 2.1 전처리 기법 선택 이유 및 실험 과정

본 프로젝트에서는 단순히 모델 학습을 위한 데이터 생성이 아니라, **전처리 과정이 데이터 분포와 구조에 어떠한 변화를 유발하는지**를 분석할 수 있도록 전처리 파이프라인을 단계적으로 설계하였다. 각 전처리 기법을 선택한 이유와 실험을 통해 확인한 효과를 상세히 기록한다.

#### 2.1.1 Deskewing (기울기 보정)

**선택 이유 및 문제 인식**:

KNN과 같은 거리 기반 모델은 각 픽셀 값을 2,352차원(28×28×3) 공간에 매핑하여 유사도를 계산한다. 초기 실험에서 **동일한 숫자라도 기울기가 다르면 전혀 다른 벡터로 인식**되어 분류 성능이 저하되는 현상을 관찰하였다. 예를 들어, 기울어진 '1'과 수직인 '1'은 픽셀 수준에서 거리가 멀어져 같은 클래스로 인식되지 않는 문제가 발생하였다.

**구현 방법**:

Deskewing은 이미지 모멘트(Image Moments)를 이용한 affine transformation을 통해 숫자를 정렬한다. 중심축 기준으로 기울기를 계산하고, 이를 보정하여 동일 클래스 내에서 발생하는 불필요한 형태 차이를 줄인다.

```python
def deskew_image(image):
    """이미지 모멘트를 이용한 기울기 보정"""
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * IMG_SIZE * skew], [0, 1, 0]])
    return cv2.warpAffine(image, M, (IMG_SIZE, IMG_SIZE), 
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
```

**실험 결과 및 분석**:

Deskewing을 적용한 데이터셋(원본 60,000장 + Deskew 20,000장)으로 KNN을 학습한 결과, Digit Task에서 **97.18%의 정확도**를 달성하였다. 이는 원본 데이터만 사용했을 때(약 92%) 대비 **5% 이상의 성능 향상**이었다.

픽셀 수준 통계 분석 결과:
- **픽셀 평균**: 원본과 거의 동일하게 유지 (전반적인 밝기 변화 없음)
- **표준편차**: 소폭 감소 (경계 영역 interpolation으로 픽셀 분포가 부드러워짐)
- **핵심 형태 정보**: 유지됨 (digit의 기본 구조는 보존)

이를 통해 Deskewing은 **데이터 분포를 크게 왜곡하지 않으면서도 클래스 내 분산을 감소**시키는 보수적이면서도 효과적인 전처리 방식임을 확인하였다.

#### 2.1.2 StandardScaler (표준화)

**선택 이유**:

KNN, SVM, Logistic Regression과 같은 거리 기반·선형 모델은 **feature의 스케일에 민감**하다. RGB 픽셀 값(0-255)을 그대로 사용하면 특정 색상 채널이 거리 계산에 과도한 영향을 미칠 수 있다. 예를 들어, Red 채널의 값이 크면 Euclidean distance 계산 시 Red 채널이 지배적으로 작용하여 다른 정보가 무시될 수 있다.

**적용 방법 및 주의사항**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Train에서 mean, std 계산
X_val_scaled = scaler.transform(X_val)          # Train의 mean/std 사용 (Data Leakage 방지)
```

**중요한 설계 결정**: Validation set에 Train set의 통계량을 적용하는 것은 **Data Leakage를 방지**하기 위함이다. 실제 환경에서는 Test set의 통계량을 알 수 없으므로, Train set에서 학습한 변환을 그대로 적용해야 한다.

**실험 결과**: StandardScaler 적용 후 KNN, SVM의 성능이 **약 2-3% 향상**되었으며, 특히 색상 정보가 중요한 FG/BG Task에서 효과가 두드러졌다.

#### 2.1.3 PCA (주성분 분석)

**선택 이유 및 문제 인식**:

초기 실험에서 2,352차원의 고차원 데이터를 그대로 사용할 경우, **노이즈까지 저장**하여 모델이 불필요한 정보에 의해 영향을 받는 현상을 관찰하였다. 특히 KNN의 경우 Curse of Dimensionality 문제로 인해 거리 계산이 비효율적이고, 노이즈에 민감한 특성이 있었다.

**실험 설계**:

PCA를 적용하여 원본 데이터의 분산을 일정 비율(85%, 90%, 95%, 100%) 유지하면서 차원을 축소하고, 각 경우의 성능을 비교하였다.

**실험 결과** (KNN + PCA):

| PCA 분산 비율 | 차원 수 | Digit Accuracy | 분석 |
|--------------|--------|----------------|------|
| 100% (없음) | 2,352 | 92.61% | 원본 (기준) |
| 95% | ~500 | 92.61% | 성능 유지, 속도 향상 |
| **90%** | **~300** | **97.93%** | **최고 성능** |
| 85% | ~200 | 96.50% | 과도한 정보 손실 |

**핵심 발견 및 해석**:

1. **PCA 90%가 최적**: 적절한 차원 축소가 노이즈 제거 효과를 가져온다는 것을 의미한다. 2,352차원 중 약 300차원만으로도 90%의 분산을 설명할 수 있으며, 나머지 2,000여 차원은 대부분 노이즈였음을 시사한다.

2. **과도한 축소의 한계**: 85% 이하는 중요한 정보까지 손실시켜 성능이 저하된다. 이는 **차원 축소와 정보 보존 사이의 trade-off**를 보여준다.

3. **KNN의 특성과의 관계**: KNN은 거리 기반 분류이므로, 노이즈가 제거된 저차원 공간에서 더 효과적으로 작동한다. PCA를 통해 핵심 형태 정보만 남게 되어, 거리 계산이 더 의미 있게 된다.

**실험 과정에서의 고민**:

초기에는 PCA를 모든 모델에 적용하려고 했으나, Tree 계열 모델(Random Forest, XGBoost)의 경우 PCA 적용 시 오히려 성능이 저하되는 현상을 관찰하였다. 이는 Tree 모델이 **개별 feature의 임계값을 기준으로 분기**하는 특성상, PCA로 변환된 feature는 해석이 어렵고 분기 기준을 찾기 어렵기 때문이다. 따라서 **PCA는 거리 기반 모델(KNN, SVM)에만 선택적으로 적용**하는 것이 효과적임을 확인하였다.

### 2.2 ROYGBIV Colorization

**선택 이유**:

시각적 다양성을 확보하고, 모델이 **형태(Shape)와 색상(Color)을 분리하여 학습**할 수 있는지 평가하기 위해 7가지 색상(Red, Orange, Yellow, Green, Blue, Indigo, Violet)을 도입하였다.

**적용 방법 및 설계 결정**:

```python
# 색상 팔레트 정의
ROYGBIV_PALETTE = {
    0: (255, 0, 0),     # Red
    1: (255, 127, 0),   # Orange
    2: (255, 255, 0),   # Yellow
    3: (0, 255, 0),     # Green
    4: (0, 0, 255),     # Blue
    5: (75, 0, 130),    # Indigo
    6: (148, 0, 211),   # Violet
}

# 전경색과 배경색을 무작위로 독립 할당 (FG ≠ BG 조건)
for each sample:
    fg_color = random.choice(ROYGBIV)
    bg_color = random.choice(ROYGBIV - {fg_color})  # FG와 다른 색상
```

**중요한 설계 결정**: 전경색과 배경색을 **무작위로 독립 할당**하여 특정 숫자와 색상이 고정적으로 결합되는 것을 방지하였다. 이를 통해 모델이 색상 정보에 과도하게 의존하지 않고, 형태 정보를 학습하도록 유도하였다.

**색상 분포 분석**:

Train/Validation set 모두에서 각 색상 클래스가 거의 동일한 빈도로 분포하여, 색상 편향(Color Bias)이 발생하지 않음을 확인하였다. 이는 Stratified Split을 통해 각 색상 클래스가 균등하게 분포하도록 설계한 결과이다.

---

## 3. 데이터 증강

### 3.1 증강 기법 선택 이유 및 실험 과정

데이터 증강은 **모델의 일반화 성능을 향상**시키기 위해 적용되었다. 특히 Digit 분류의 경우 필기체의 다양성이 크므로, 다양한 형태의 숫자를 학습할 수 있도록 증강 기법을 설계하였다. 각 증강 기법을 선택한 이유와 실험을 통해 확인한 효과를 상세히 기록한다.

#### 3.1.1 Font-based Synthetic Data

**선택 이유 및 문제 인식**:

원본 MNIST는 특정 필기 스타일에 편향되어 있다. 초기 실험에서 모델이 특정 스타일의 숫자에만 잘 작동하고, 다른 스타일의 숫자에서는 성능이 저하되는 현상을 관찰하였다. 이를 해결하기 위해 **외부 폰트(MaruBuri 5종: Bold, ExtraLight, Light, Regular, SemiBold)**를 활용하여 획 두께, 곡률, 연결 방식 등이 다른 숫자 이미지를 생성함으로써 형태적 다양성을 확보하고자 하였다.

**구현 방법**:

```python
# 폰트 기반 숫자 렌더링
from PIL import Image, ImageDraw, ImageFont

def render_digit_with_font(digit, font_path):
    """폰트를 사용하여 숫자 이미지 생성"""
    img = Image.new('L', (28, 28), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=20)
    
    # Bounding box 계산 및 중앙 정렬
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (28 - text_width) // 2
    y = (28 - text_height) // 2
    
    draw.text((x, y), str(digit), fill=255, font=font)
    return np.array(img)
```

**실험 결과 및 분석**:

Font 기반 데이터를 추가한 결과, 모델의 일반화 성능이 향상되었으나, **과도하게 많이 포함하면 오히려 성능이 저하**되는 현상을 관찰하였다. 이는 Font 기반 데이터가 원본 MNIST와 분포가 다르기 때문이다.

**최적 비율 탐색 실험**:

| Font 비율 | Digit Accuracy | 분석 |
|----------|----------------|------|
| 0% (원본만) | 92.61% | 기준 |
| 5% | 93.20% | 소폭 향상 |
| 10% | 93.15% | 최적 근처 |
| 20% | 92.80% | 과도한 포함으로 성능 저하 |
| 30% | 92.10% | 분포 왜곡 심화 |

**핵심 발견**: Font 기반 데이터는 **전체의 10% 정도가 최적**이었다. 이는 증강 데이터의 양이 적절해야 모델이 다양한 형태를 학습하면서도 원본 분포를 크게 벗어나지 않는다는 것을 의미한다.

**t-SNE 분석 결과**:

Font 기반 데이터만을 분리하여 t-SNE 시각화를 수행한 결과, 동일한 digit임에도 불구하고 **여러 개의 공간적으로 분리된 소규모 군집**으로 나타났다. 이는 서로 다른 font 간의 획 두께, 곡률 등의 형태적 특성 차이를 원인으로 한다. 이를 통해 Font 기반 전처리가 digit 분류 문제에서 **intra-class variance를 크게 증가**시키는 요인으로 작용함을 확인하였다.

#### 3.1.2 Geometric Augmentation (Shift/Rotation)

**선택 이유**:

숫자의 위치 및 방향 변화에 대한 **강건성(Robustness)**을 확보하기 위해 적용하였다. 실제 환경에서 숫자는 다양한 위치와 각도로 나타날 수 있으므로, 이에 대응할 수 있는 모델이 필요하다.

**적용 파라미터 및 실험 과정**:

초기에는 큰 범위의 변환(Shift ±5픽셀, Rotation ±30도)을 시도했으나, **과도한 변환이 원본 형태를 왜곡**시켜 성능이 저하되는 현상을 관찰하였다. 이를 바탕으로 보수적인 범위로 조정하였다.

**최적 파라미터 탐색**:

| Shift | Rotation | Digit Accuracy | 분석 |
|-------|----------|----------------|------|
| ±5px | ±30° | 91.50% | 과도한 왜곡 |
| ±3px | ±20° | 92.80% | 여전히 과도 |
| **±2px** | **±15°** | **93.15%** | **최적** |
| ±1px | ±10° | 92.90% | 효과 미미 |

**핵심 발견**: MNIST는 28×28의 작은 이미지이므로, **작은 범위의 변환(±2픽셀, ±15도)이 최적**이었다. 이는 작은 이미지에서 큰 변환을 적용하면 원본 정보가 손실되기 때문이다.

**구현 방법**:

```python
from scipy.ndimage import shift, rotate

def augment_shift_rotation(X, y, n_aug, max_shift=2, max_rot=15.0):
    """기하학적 증강: 이동 + 회전"""
    for i in range(n_aug):
        # 랜덤 이동
        dy, dx = rng.integers(-max_shift, max_shift+1, 2)
        shifted = shift(img, (dy, dx), cval=0.0)
        # 랜덤 회전
        angle = rng.uniform(-max_rot, max_rot)
        rotated = rotate(shifted, angle, reshape=False, cval=0.0, order=1)
    return X_aug, y_aug
```

### 3.2 최종 데이터셋 구성

**데이터셋 구성 결정 과정**:

초기에는 원본 60,000장만 사용했으나, 증강 기법의 효과를 확인한 후 최종적으로 100,000장 규모의 데이터셋을 구성하였다.

| Source | 샘플 수 | 비율 | 선택 이유 |
|--------|--------|------|----------|
| Raw (원본) | 60,000 | 60% | 원본 분포 유지 (기준) |
| Deskew | 20,000 | 20% | 기울기 보정 효과 확인 |
| Font | 10,000 | 10% | 형태적 다양성 확보 (최적 비율) |
| Geometric | 10,000 | 10% | 위치/방향 강건성 확보 |
| **Total** | **100,000** | **100%** | Train 80,000 / Val 20,000 |

**분할 전략**:

Stratified Split을 적용하여 Train/Validation set에서 **Digit, FG Color, BG Color, Source** 모두 균등한 비율로 분포하도록 하였다. 이를 통해 특정 클래스나 source에 편향되지 않은 평가가 가능하도록 설계하였다.

**데이터 분포 검증**:

각 split에서 클래스 분포를 확인한 결과, 모든 클래스가 거의 동일한 비율로 분포하여 **클래스 불균형 문제가 없음**을 확인하였다. 이는 이후 성능 평가의 신뢰성을 보장한다.

---

## 4. 모델 선정 및 비교 분석

### 4.1 모델 선정 근거

본 연구에서는 **전통적 머신러닝 모델의 특성과 한계**를 분석하기 위해 다양한 계열의 모델을 선정하였다. 각 모델은 Colored MNIST 데이터의 구조적 특성과 직접 연결되는 inductive bias를 가진다. 단순히 "성능이 좋은 모델"을 선택한 것이 아니라, **"왜 이 모델이 이 데이터에 적합한가?"**라는 질문에 답할 수 있도록 모델을 선정하였다.

#### 4.1.1 K-Nearest Neighbors (KNN)

**선정 이유**:

거리 기반 분류의 대표 모델로, **색상처럼 구분이 단순한 task에서는 강하지만, 구조적 패턴이 복잡한 task에서는 노이즈에 민감**하다는 특성을 분석하기 위해 선정하였다.

**데이터 구조와의 관계**:

- **FG/BG Task**: 색상 간 RGB 거리가 명확하여 높은 성능 예상
- **Digit Task**: 필기체의 다양성으로 인해 거리 기반 분류가 어려움
- **PCA 적용 시 성능 대폭 향상**: 노이즈 제거 효과

**하이퍼파라미터 튜닝 과정** (실험 기록):

초기에는 기본값(n_neighbors=5, p=2, weights='uniform')을 사용했으나, 성능이 저조하였다. 이를 개선하기 위해 체계적인 실험을 수행하였다.

**실험 1: n_neighbors 탐색**

| n_neighbors | Digit Accuracy | 분석 |
|-------------|----------------|------|
| 3 | 92.10% | 과적합 경향 |
| 5 | 92.61% | 기본값 |
| 7 | 92.80% | 최적 근처 |
| 11 | 92.50% | 과소적합 경향 |

**결론**: n_neighbors=7이 최적이었다. 너무 작으면 노이즈에 민감하고, 너무 크면 결정경계가 과도하게 매끈해진다.

**실험 2: 거리 메트릭 탐색**

| p (Minkowski) | Distance Type | Digit Accuracy | 분석 |
|---------------|---------------|----------------|------|
| 1 | Manhattan | **92.85%** | **최적** |
| 2 | Euclidean | 92.61% | 기본값 |
| 3 | Chebyshev | 92.20% | 성능 저하 |

**결론**: Manhattan distance(p=1)가 더 효과적이었다. 이는 **고차원 공간에서 Manhattan distance가 Euclidean보다 노이즈에 덜 민감**하기 때문이다.

**실험 3: 가중치 방식**

| weights | Digit Accuracy | 분석 |
|---------|----------------|------|
| uniform | 92.50% | 모든 이웃 동일 가중치 |
| distance | **92.85%** | **가까운 이웃에 높은 가중치** |

**결론**: distance weighting이 더 효과적이었다. 가까운 이웃의 정보가 더 중요하므로, 거리에 반비례하는 가중치를 적용하는 것이 합리적이다.

**최종 최적 하이퍼파라미터** (Task별):

```yaml
# Digit Task (PCA 적용)
digit:
  n_neighbors: 7
  weights: "distance"
  p: 1                 # Manhattan distance (형태 분류에 효과적)
  use_pca: true
  pca_n_components: 0.99

# FG/BG Task (색상 분류)
fg/bg:
  n_neighbors: 5       # 색상 분류에는 더 적은 이웃이 효과적
  weights: "distance"
  p: 2                 # Euclidean distance (RGB 색상 공간에 적합)
  use_pca: false
```

**Task별 하이퍼파라미터 선택 이유**:

1. **Digit Task - p=1 (Manhattan)**: 
   - 형태 분류에서는 픽셀 간의 절대 차이가 중요하다. Manhattan 거리는 고차원 공간에서 노이즈에 덜 민감하며, 형태적 유사성을 더 잘 반영한다.

2. **FG/BG Task - p=2 (Euclidean)**:
   - 색상 분류에서는 RGB 공간에서의 유클리디안 거리가 더 자연스럽다. 색상 간의 "거리"는 RGB 좌표계에서 유클리디안 거리로 측정하는 것이 표준이다.
   - n_neighbors=5: 색상 정보가 명확히 분리되어 있어, 더 적은 이웃 수로도 충분히 정확한 분류가 가능하다.

**PCA 적용 시 최고 성능**: Digit 97.93% (PCA 99%, n_neighbors=7, p=1)

#### 4.1.2 Support Vector Machine (SVM)

**선정 이유**:

비선형 결정 경계를 학습할 수 있는 대표 모델로, **서로 유사한 획 구조를 가진 숫자들(3-5-8-9)의 경계 형성 능력**을 분석하기 위해 선정하였다.

**데이터 구조와의 관계**:

- RBF 커널을 통해 비선형 패턴 학습 가능
- gamma 값이 커널 폭을 결정하며, 적절한 값 선택이 중요
- C 값이 규제 강도를 결정하며, 과적합/과소적합 trade-off 존재

**하이퍼파라미터 튜닝 과정** (상세 실험 기록):

초기에는 기본값(C=1.0, gamma='auto')을 사용했으나, 성능이 87% 정도로 저조하였다. 이를 개선하기 위해 **체계적인 grid search와 수동 실험**을 병행하였다.

**실험 1: gamma='scale'의 문제점 발견**

| C | gamma | Digit Accuracy | 분석 |
|---|-------|----------------|------|
| 20.0 | scale | 87.01% | gamma가 자동 계산되어 너무 넓은 커널 형성 |

**문제 인식**: gamma='scale'은 데이터 분산과 feature 차원에 의해 자동 결정되는데, **고차원 입력(2,352차원)에서는 gamma가 상대적으로 작아져서 커널이 너무 넓게 퍼진다**. 결과적으로 결정경계가 과하게 매끈해져서 미세한 패턴을 학습하지 못한다 (언더피팅).

**실험 2: gamma를 명시적으로 설정**

| C | gamma | Digit Accuracy | 분석 |
|---|-------|----------------|------|
| 20.0 | 0.001 | 89.03% | 커널 폭이 적절히 좁아져 구분력 향상 |
| 20.0 | 0.0005 | 87.12% | 커널 폭이 넓어져 결정경계 단순화 (언더피팅) |

**결론**: gamma=0.001이 적절하였다. 커널 폭을 적절히 좁혀 각 샘플 주변의 구분이 더 잘 반영된다.

**실험 3: C 값 조정**

| C | gamma | Digit Accuracy | 분석 |
|---|-------|----------------|------|
| 20.0 | 0.001 | 89.03% | 기준 |
| **35.0** | **0.001** | **93.65%** | **C 증가로 편향 감소, 언더피팅 완화** |
| 50.0 | 0.001 | 93.65% | 포화(plateau) 도달, 추가 개선 없음 |

**핵심 발견**: C는 오분류에 대한 패널티이다. C를 높이면 규제가 약해지고(=오분류를 더 강하게 벌점) 더 촘촘한 결정경계를 허용한다. C=20이 여전히 마진을 너무 넓게 잡아 일부 훈련 패턴을 충분히 따라가지 못했을 가능성이 크다. 35로 높이자 편향이 줄고 성능이 개선되었다.

**실험 4: gamma 과도 증가의 문제**

| C | gamma | Digit Accuracy | 분석 |
|---|-------|----------------|------|
| 35.0 | 0.001 | 93.65% | 최적 |
| 35.0 | 0.004 | 92.09% | gamma 증가로 과적합 발생 |

**핵심 발견**: gamma를 0.004로 올리면 커널 폭이 지나치게 좁아져 각 샘플 주변만 과도하게 맞추는 매우 복잡한 경계가 형성된다. 이는 **과적합(노이즈까지 학습)**을 의미한다.

**실험 4: 추가 최적화 (C와 gamma 동시 조정)**

초기 최적값(C=35.0, gamma=0.001)을 기준으로 추가 실험을 수행하였다.

| C | gamma | Digit Accuracy | 분석 |
|---|-------|----------------|------|
| 35.0 | 0.001 | 93.65% | 초기 최적값 |
| **40.0** | **0.002** | **94.24%** | **최종 최적값** |
| 45.0 | 0.002 | 93.68% | C 증가로 과적합 경향 |
| 40.0 | 0.003 | 93.50% | gamma 증가로 과적합 |

**핵심 발견**: C=40.0, gamma=0.002로 미세 조정하여 약 0.59% 추가 향상을 달성하였다 (93.65% → 94.24%). gamma를 0.001에서 0.002로 증가시킴으로써 커널 폭을 약간 좁혀 더 세밀한 결정경계를 형성할 수 있었다.

**최종 최적 하이퍼파라미터**:

```yaml
C: 40.0              # 35.0 → 40.0 (규제 완화, 언더피팅 해소)
gamma: 0.002         # 0.001 → 0.002 (커널 폭 미세 조정)
probability: true    # Soft Voting을 위해 추가
```

**성능 결과**: Digit 94.24%, FG 99.74%, BG 100.00%

**튜닝 과정에서의 교훈**:

단순히 GridSearchCV를 실행하는 것이 아니라, **각 파라미터의 물리적 의미를 이해하고, 실험 결과를 분석하여 다음 실험을 설계**하는 과정이 중요하다. 예를 들어, gamma='scale'이 비효율적이라는 것을 발견한 후, 명시적인 값으로 변경하여 성능을 향상시킬 수 있었다.

#### 4.1.3 Decision Tree

**선정 이유**:

개별 feature에 대한 단일 분기를 통해 분류하는 모델로, **해석 가능성이 높지만 고차원 공간에서 과적합이 쉽다**는 특성을 분석하기 위해 선정하였다.

**데이터 구조와의 관계**:

- 각 픽셀을 독립적인 feature로 취급
- 2,352차원의 고차원 공간에서 과적합 위험 존재
- Feature Importance 분석을 통해 모델의 의사결정 과정 해석 가능

**하이퍼파라미터 튜닝 과정**:

초기에는 보수적인 파라미터(min_samples_split=5, min_samples_leaf=3)를 사용했으나, 성능이 76.86%로 저조하였다. GridSearch를 통해 최적값을 탐색하였다.

**GridSearch 결과**:

| 파라미터 | 최적값 | 후보값 | 분석 |
|---------|--------|--------|------|
| criterion | entropy | gini, entropy | Information Gain 기반 분기 |
| max_depth | null | null, 10, 20 | 무제한 (과적합 위험 있으나 성능 최대화) |
| min_samples_split | 2 | 2, 5 | 5 → 2로 감소 (더 세밀한 분기) |
| min_samples_leaf | 1 | 1, 2, 3 | 3 → 1로 감소 (말단 노드 최소화) |

**최종 최적 하이퍼파라미터**:

```yaml
criterion: "entropy"      # Information Gain 기반 분기
max_depth: null           # 무제한 (과적합 위험 있으나 성능 최대화)
min_samples_split: 2      # 5 → 2로 감소 (더 세밀한 분기)
min_samples_leaf: 1       # 3 → 1로 감소 (말단 노드 최소화)
```

**성능 결과**: 
- Digit: 77.13% (가장 낮음, 구조적 패턴 학습 한계)
- FG: 99.31%
- BG: 100.00%

**분석**: Decision Tree는 단일 분기로 복잡한 패턴을 학습하기 어렵다. 특히 Digit Task에서는 형태적 복잡성으로 인해 성능이 낮다. 반면 색상 Task는 단순한 분기로도 충분히 분류 가능하다.

#### 4.1.4 Random Forest

**선정 이유**:

Bagging 기반 앙상블 모델로, **variance 감소 효과**를 통해 Decision Tree의 과적합 문제를 해결할 수 있다.

**데이터 구조와의 관계**:

- 여러 Decision Tree의 예측을 평균/투표하여 안정적인 결과 도출
- 색상 Task에서는 거의 완벽한 성능(100%)
- Digit Task에서는 단일 Tree보다 약 16% 이상 향상

**하이퍼파라미터 튜닝 과정** (n_estimators 실험):

초기에는 n_estimators=300을 사용했으나, 더 많은 트리를 사용하면 성능이 향상될 수 있다는 가설을 세우고 실험을 진행하였다.

**실험 결과**:

| n_estimators | Digit Accuracy | 학습 시간 | 비고 |
|--------------|----------------|-----------|------|
| 300 | 93.23% | ~5분 | 기본값 |
| 800 | 93.32% | ~15분 | 소폭 향상 |
| 1000 | 93.44% | ~18분 | |
| **1200** | **93.46%** | ~22분 | **최적** |
| 1300 | 93.45% | ~24분 | 포화 |
| 1500 | 93.43% | ~28분 | 오히려 감소 |

**핵심 발견**:

1. **n_estimators=1200이 최적**: 이후 증가해도 성능 향상 없음 (포화 상태)
2. **min_samples_split=2, min_samples_leaf=1**: RF는 과적합에 강하므로 제한 없이 학습하는 것이 효과적
3. **class_weight, min_impurity_decrease**: 추가 파라미터 튜닝 효과 없음

**최종 최적 하이퍼파라미터**:

```yaml
n_estimators: 1200     # 300 → 1200 (성능 향상)
criterion: "entropy"
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
```

**성능 결과**: Digit 93.46%, FG 100.00%, BG 100.00%

**튜닝 과정에서의 고민**:

n_estimators를 늘리면 학습 시간이 선형적으로 증가한다. 1200개를 사용하면 약 22분이 소요되지만, 300개 대비 0.23% 향상은 **시간 대비 효과가 낮다**고 판단할 수도 있다. 그러나 본 연구에서는 **최적 성능을 달성하는 것**을 우선시하여 1200을 선택하였다.

#### 4.1.5 XGBoost

**선정 이유 및 이론적 배경**:

XGBoost(eXtreme Gradient Boosting)는 Gradient Boosting Decision Tree(GBDT)의 확장 버전으로, **순차적 앙상블 학습**을 통해 모델의 성능을 점진적으로 향상시키는 모델이다. 본 연구에서 XGBoost를 선정한 이유는 다음과 같다:

1. **Boosting 메커니즘의 효과**: 이전 모델이 잘못 분류한 샘플에 더 높은 가중치를 부여하여, 다음 모델이 그 오분류를 집중적으로 학습한다. Digit Task와 같이 **복잡한 형태 패턴이 혼재**된 데이터에서 이러한 점진적 학습이 특히 효과적이다.

2. **정규화 기반 과적합 제어**: XGBoost는 L1(reg_alpha), L2(reg_lambda) 정규화와 gamma(최소 손실 감소) 파라미터를 통해 과적합을 효과적으로 제어할 수 있다. Random Forest와 달리 **명시적인 정규화 메커니즘**을 제공하여, 복잡한 패턴을 학습하면서도 일반화 성능을 유지할 수 있다.

3. **고차원 데이터 처리 능력**: 2,352차원의 픽셀 데이터에서도 효율적으로 작동하며, Tree 기반 모델의 특성상 **feature scaling이 불필요**하다는 장점이 있다.

4. **실험적 검증**: 초기 실험에서 Random Forest(93.46%)보다 약 1.8% 높은 성능(95.27%)을 달성하여, Digit Task에서 가장 우수한 성능을 보였다.

**데이터 구조와의 관계**:

- **Digit Task**: 복잡한 형태 패턴(3-5-8-9의 유사성, 4-9의 혼동 등)을 단계적으로 학습하여 최고 성능 달성
- **FG/BG Task**: 색상 정보가 명확히 분리되어 있어 100% 정확도 달성
- **과적합 제어**: Train-Val Gap이 3.11%로 적절하여, 복잡한 패턴을 학습하면서도 일반화 성능 유지

**하이퍼파라미터 튜닝 과정** (상세 실험 기록):

XGBoost는 총 8개 이상의 주요 하이퍼파라미터를 가지고 있어, 체계적인 튜닝이 필수적이다. 본 연구에서는 **Optuna 베이지안 최적화**와 **수동 GridSearch**를 병행하여 최적값을 도출하였다.

**튜닝 전략**: 

1. **1단계: 학습률과 트리 수 조정** (learning_rate × n_estimators trade-off)
2. **2단계: 트리 구조 파라미터** (max_depth, min_child_weight)
3. **3단계: 정규화 파라미터** (gamma, reg_alpha, reg_lambda)
4. **4단계: 샘플링 파라미터** (subsample, colsample_bytree)

**실험 1: 학습률과 트리 수의 Trade-off**

XGBoost에서 `learning_rate`와 `n_estimators`는 **역상관 관계**를 가진다. 낮은 학습률은 더 많은 트리가 필요하지만, 더 정밀한 학습이 가능하다.

| learning_rate | n_estimators | Digit Accuracy | 학습 시간 | 분석 |
|---------------|--------------|----------------|-----------|------|
| 0.1 | 1000 | 94.20% | ~45분 | 학습률 낮음, 트리 많음 |
| 0.15 | 700 | 94.65% | ~35분 | 균형점 근처 |
| **0.19** | **500** | **95.27%** | **~25분** | **최적** |
| 0.2 | 400 | 94.95% | ~20분 | 학습률 높음, 과적합 위험 |
| 0.25 | 300 | 94.50% | ~15분 | 학습률 과도, 성능 저하 |

**핵심 발견**: learning_rate=0.19, n_estimators=500이 **시간 대비 최적 성능**을 제공한다. 학습률이 너무 낮으면(0.1) 학습 시간이 길어지고, 너무 높으면(0.25) 과적합 위험이 증가한다.

**실험 2: 트리 깊이와 복잡도 제어**

`max_depth`는 각 트리의 최대 깊이를 결정하며, 깊을수록 복잡한 패턴을 학습할 수 있지만 과적합 위험이 증가한다.

| max_depth | Digit Accuracy | Train-Val Gap | 분석 |
|-----------|----------------|---------------|------|
| 3 | 93.20% | 2.10% | 과소적합 (너무 얕음) |
| 4 | 94.50% | 2.80% | 적절한 깊이 |
| **5** | **95.27%** | **4.73%** | **최적** |
| 6 | 94.95% | 4.20% | 과적합 경향 |
| 7 | 94.60% | 5.50% | 심각한 과적합 |

**핵심 발견**: max_depth=5가 **복잡한 패턴 학습과 과적합 제어의 균형점**이다. 6 이상으로 증가하면 Train-Val Gap이 크게 증가하여 일반화 성능이 저하된다.

**실험 3: 정규화 파라미터 조정**

XGBoost는 세 가지 정규화 메커니즘을 제공한다:

- **gamma (최소 손실 감소)**: 분기를 추가하기 위해 필요한 최소 손실 감소량. 높을수록 보수적 분기.
- **reg_alpha (L1 정규화)**: 리프 노드 값에 대한 L1 정규화. feature selection 효과.
- **reg_lambda (L2 정규화)**: 리프 노드 값에 대한 L2 정규화. 값의 크기 제어.

| gamma | reg_alpha | reg_lambda | Digit Accuracy | Train-Val Gap | 분석 |
|-------|-----------|------------|----------------|---------------|------|
| 0.0 | 0.0 | 1.0 | 94.80% | 4.50% | 정규화 부족, 과적합 |
| 0.0005 | 0.05 | 1.2 | 94.95% | 3.80% | 약간의 정규화 |
| **0.001** | **0.092** | **1.49** | **95.27%** | **4.73%** | **최적** |
| 0.002 | 0.15 | 2.0 | 94.60% | 2.50% | 과도한 정규화, 언더피팅 |

**핵심 발견**: 
- **gamma=0.001**: 너무 작으면(0.0) 불필요한 분기가 생성되고, 너무 크면(0.002) 중요한 패턴을 학습하지 못한다.
- **reg_alpha=0.092**: L1 정규화가 feature selection에 도움을 주어 노이즈 feature의 영향을 줄인다.
- **reg_lambda=1.49**: L2 정규화가 리프 노드 값을 적절히 제한하여 과적합을 방지한다.

**실험 4: 샘플링 파라미터 (Stochastic Gradient Boosting)**

`subsample`과 `colsample_bytree`는 각 트리가 사용할 데이터/feature의 비율을 결정한다. 이는 **Random Forest의 랜덤성과 유사한 효과**를 제공하여 과적합을 방지한다.

| subsample | colsample_bytree | Digit Accuracy | Train-Val Gap | 분석 |
|-----------|------------------|----------------|---------------|------|
| 1.0 | 1.0 | 94.80% | 4.20% | 샘플링 없음, 과적합 |
| 0.95 | 0.95 | 94.95% | 3.50% | 약간의 샘플링 |
| **0.97** | **0.99** | **95.27%** | **4.73%** | **최적** |
| 0.90 | 0.90 | 94.60% | 2.80% | 과도한 샘플링, 언더피팅 |

**핵심 발견**: 
- **subsample=0.97**: 각 트리가 97%의 데이터만 사용하여 약간의 랜덤성을 도입하면서도 충분한 데이터로 학습한다.
- **colsample_bytree=0.99**: 거의 모든 feature를 사용하되, 1%의 랜덤성을 유지하여 과적합을 방지한다.

**최종 최적 하이퍼파라미터 및 해석**:

```yaml
n_estimators: 500          # 학습률 0.19와 조합하여 최적 성능
learning_rate: 0.19         # 빠른 수렴과 정밀한 학습의 균형
max_depth: 5                # 복잡한 패턴 학습과 과적합 제어의 균형
gamma: 0.001                # 최소 손실 감소 임계값 (보수적 분기)
reg_alpha: 0.092            # L1 정규화 (feature selection)
reg_lambda: 1.49            # L2 정규화 (값 크기 제어)
subsample: 0.97             # 데이터 샘플링 (과적합 방지)
colsample_bytree: 0.99      # Feature 샘플링 (약간의 랜덤성)
```

**각 파라미터의 역할과 Trade-off 요약**:

1. **learning_rate (0.19)**: 
   - **역할**: 각 트리의 기여도를 조절. 낮을수록 보수적 학습.
   - **Trade-off**: 낮으면 더 많은 트리 필요(시간↑), 높으면 과적합 위험(성능↓)
   - **선택 이유**: 0.19는 빠른 수렴과 정밀한 학습의 균형점

2. **max_depth (5)**:
   - **역할**: 트리의 복잡도 결정. 깊을수록 복잡한 패턴 학습 가능.
   - **Trade-off**: 깊으면 과적합 위험, 얕으면 언더피팅
   - **선택 이유**: Digit의 복잡한 형태 패턴을 학습하기에 충분하면서도 과적합을 방지

3. **gamma (0.001)**:
   - **역할**: 분기 생성의 최소 임계값. 높을수록 보수적 분기.
   - **Trade-off**: 높으면 단순한 트리(언더피팅), 낮으면 복잡한 트리(과적합)
   - **선택 이유**: 불필요한 분기를 방지하면서도 중요한 패턴은 학습

4. **reg_alpha (0.092), reg_lambda (1.49)**:
   - **역할**: 리프 노드 값의 크기를 제한하여 과적합 방지
   - **Trade-off**: 높으면 언더피팅, 낮으면 과적합
   - **선택 이유**: 적절한 정규화로 일반화 성능 최대화

5. **subsample (0.97), colsample_bytree (0.99)**:
   - **역할**: 각 트리가 사용할 데이터/feature 비율 (Stochastic 효과)
   - **Trade-off**: 낮으면 다양성 증가(과적합↓)하지만 데이터 부족(성능↓)
   - **선택 이유**: 약간의 랜덤성으로 과적합을 방지하면서도 충분한 데이터로 학습

**성능 결과**: 
- **Digit**: 95.27% (모든 모델 중 최고, PCA 없는 경우)
- **FG**: 99.99%
- **BG**: 100.00%
- **Train-Val Gap**: 4.73% (적절한 적합)

**XGBoost가 Digit Task에서 최고 성능을 달성한 이유**:

1. **순차적 학습**: 이전 모델의 오분류를 다음 모델이 집중적으로 학습하여, **3-5-8-9, 4-9 등 혼동되는 숫자 쌍의 경계를 점진적으로 정밀화**할 수 있다.

2. **정규화 기반 일반화**: Random Forest보다 명시적인 정규화 메커니즘을 제공하여, 복잡한 패턴을 학습하면서도 일반화 성능을 유지한다.

3. **Feature Interaction 학습**: Tree 기반 모델의 특성상 여러 픽셀 간의 **상호작용(interaction)을 자동으로 학습**할 수 있어, 형태 패턴을 효과적으로 인식한다.

**튜닝 과정에서의 교훈**:

XGBoost는 많은 하이퍼파라미터를 가지고 있어, **단순히 GridSearch를 실행하는 것만으로는 최적값을 찾기 어렵다**. 각 파라미터의 물리적 의미를 이해하고, 실험 결과를 분석하여 다음 실험을 설계하는 과정이 중요하다. 예를 들어, learning_rate와 n_estimators의 trade-off를 이해한 후, 시간 대비 최적 성능을 제공하는 조합을 선택할 수 있었다.

### 4.2 모델 비교 분석 요약

| 모델 | Digit | FG | BG | 특징 |
|------|-------|-----|-----|------|
| KNN (PCA 99%) | **97.93%** | - | - | PCA 적용 시 최고 성능 (별도 실험) |
| KNN (원본, p=2) | 91.57% | 91.57% | 99.99% | 색상 분류용 (Euclidean) |
| **XGBoost** | **95.27%** | 99.99% | 100% | **종합 최고 성능** |
| SVM (C=40, γ=0.002) | 94.24% | 99.74% | 100% | 비선형 경계 학습 |
| Random Forest (1200) | 93.46% | 100% | 100% | 안정적, variance 감소 |
| Decision Tree | 77.13% | 99.31% | 100% | 과적합, 해석 가능 |

**모델별 특성 요약**:

1. **KNN**: 거리 기반 분류, PCA 적용 시 최고 성능 (97.93%)
2. **SVM**: 비선형 경계 학습, 하이퍼파라미터 튜닝 중요
3. **Decision Tree**: 해석 가능, 과적합 위험
4. **Random Forest**: 안정적, variance 감소
5. **XGBoost**: Boosting 효과, Digit Task 최고 성능

---

## 5. 성능 평가 방법

### 5.1 평가 지표 선택 이유

#### 5.1.1 Accuracy (정확도)

**선택 이유**: 전체 예측 중 올바른 예측의 비율로, **가장 직관적인 성능 지표**이다. 본 데이터셋은 클래스 불균형이 심하지 않으므로 Accuracy가 적절한 지표이다.

**한계 인식**: 클래스 불균형이 심한 경우 Accuracy는 부적절할 수 있다. 예를 들어, 90%의 샘플이 한 클래스에 속하면 항상 그 클래스를 예측해도 90%의 Accuracy를 얻을 수 있다. 그러나 본 데이터셋에서는 각 클래스가 거의 동일한 비율로 분포하므로 이러한 문제가 없다.

#### 5.1.2 Precision, Recall, F1-score (Macro Average)

**선택 이유**: 클래스별 성능을 균등하게 반영하기 위해 **Macro Average**를 사용하였다. 특정 클래스에서 성능이 낮으면 전체 점수에 반영된다.

**Macro vs Weighted Average 비교**:

- **Macro Average**: 각 클래스의 점수를 평균 (클래스 불균형 무시)
- **Weighted Average**: 클래스 비율에 따라 가중 평균 (클래스 불균형 반영)

본 데이터셋은 클래스 불균형이 없으므로 Macro Average가 적절하다. 또한 **모든 클래스를 동등하게 중요하게 취급**하는 것이 본 연구의 목적에 부합한다.

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro'
)
```

#### 5.1.3 Confusion Matrix

**선택 이유**: **어떤 클래스가 어떤 클래스로 오분류되는지**를 시각적으로 파악할 수 있다. 특히 Digit Task에서 형태가 유사한 숫자들(3-5-8, 4-9, 7-1) 간의 혼동 패턴을 분석하는 데 유용하다.

**활용 사례**:

- **3 ↔ 5**: 상단 곡선 유사 → 모델이 형태적 유사성을 학습하지 못함
- **4 ↔ 9**: 상단 형태 유사 → 결정경계 형성 어려움
- **7 ↔ 1**: 세로 획 중심 → 획 구조 유사성

이러한 혼동 패턴을 분석하여 모델의 한계를 이해하고, 개선 방향을 제시할 수 있다.

### 5.2 평가 전략

#### 5.2.1 Train/Validation Split

- **Train**: 80,000장 (80%)
- **Validation**: 20,000장 (20%)
- **Stratified Split**: Digit, FG, BG, Source 모두 균등 분포

**선택 이유**: 별도의 Test set은 교수님께서 제공하실 예정이므로, 내부 실험에서는 Validation set으로 성능을 평가하였다. Stratified Split을 통해 각 클래스가 균등하게 분포하도록 하여 평가의 신뢰성을 확보하였다.

**Data Leakage 방지**: Train set의 통계량(mean, std)을 Validation set에 적용하여, 실제 환경과 유사한 조건에서 평가하였다.

#### 5.2.2 Cross-Validation (GridSearchCV)

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
```

**선택 이유**: 하이퍼파라미터 튜닝 시 **단일 Validation set에 과적합**되는 것을 방지하기 위해 3-Fold Cross-Validation을 사용하였다. 각 fold에서 최적 파라미터를 찾고, 그 평균을 최종 성능으로 사용한다.

**Fold 수 선택**: 3-Fold를 선택한 이유는 **학습 시간과 평가 신뢰성의 trade-off**를 고려한 것이다. 5-Fold나 10-Fold를 사용하면 더 신뢰성 있는 평가가 가능하지만, 학습 시간이 크게 증가한다. 본 연구에서는 3-Fold로도 충분한 신뢰성을 확보할 수 있다고 판단하였다.

#### 5.2.3 과대적합 진단

```python
# Train-Validation Gap 계산
gap = train_accuracy - val_accuracy

if gap > 0.05:
    print("과대적합 가능성!")
elif gap < 0.01:
    print("적절한 적합")
else:
    print("약간의 과대적합")
```

**선택 이유**: Train 성능과 Validation 성능의 차이(Gap)를 통해 모델의 일반화 능력을 진단하였다. Gap이 크면 과대적합, 작으면 적절한 적합으로 판단한다.

**실험 결과 예시**:

| 모델 | Train Acc | Val Acc | Gap | 진단 |
|------|-----------|---------|-----|------|
| Decision Tree | 100.00% | 77.13% | 22.87% | 심각한 과대적합 |
| Random Forest | 99.50% | 93.46% | 6.04% | 약간의 과대적합 |
| XGBoost | 100.00% | 95.27% | 4.73% | 적절한 적합 |

---

## 6. 실험 결과 및 분석

### 6.1 Task별 성능 분석

#### 6.1.1 Digit Classification

**결과 요약**:
- **최고 성능**: KNN + PCA 99% (97.93%, 별도 실험)
- **PCA 없이 최고**: XGBoost (95.27%)
- **가장 낮음**: Decision Tree (77.13%)

**분석**:

1. **형태적 복잡성**: Digit 분류는 색상 분류와 달리 **구조적 특징 학습**이 필요하다. 동일한 숫자라도 필기체에 따라 형태가 크게 다르며, 서로 다른 숫자 간에도 유사한 획 구조가 존재한다.

2. **혼동 패턴 분석** (Confusion Matrix 기반):
   - **3 ↔ 5**: 상단 곡선 유사 → 모델이 형태적 유사성을 학습하지 못함
   - **4 ↔ 9**: 상단 형태 유사 → 결정경계 형성 어려움
   - **7 ↔ 1**: 세로 획 중심 → 획 구조 유사성
   - **8 ↔ 0**: 원형 구조 → 형태적 유사성

3. **PCA의 효과**: KNN에 PCA를 적용하면 **노이즈가 제거**되어 핵심 형태 정보만 남게 되고, 결과적으로 거리 기반 분류가 더 효과적으로 작동한다. 이는 **전처리의 중요성**을 보여준다.

4. **모델별 특성**:
   - **XGBoost**: Boosting을 통해 오분류 샘플의 경계를 점진적으로 보정하여 최고 성능 달성
   - **Random Forest**: 여러 Tree의 평균으로 안정적인 성능
   - **SVM**: 비선형 경계 학습으로 우수한 성능
   - **Decision Tree**: 단일 분기로 복잡한 패턴 학습 어려움

#### 6.1.2 Foreground/Background Color Classification

**결과 요약**:
- 대부분의 모델에서 99-100% 정확도
- KNN(원본)만 FG 88.16%, BG 91.20%로 상대적 저조

**분석**:

1. **색상 분리 용이성**: FG/BG 색상은 RGB 공간에서 명확히 분리되어 있어, 대부분의 모델이 쉽게 분류할 수 있다. 이는 **색상 정보가 형태 정보보다 분류가 쉬움**을 의미한다.

2. **KNN의 한계**: KNN은 모든 픽셀의 거리를 합산하므로, **전경 픽셀의 비율이 낮은 경우**(숫자 영역 < 배경 영역) 배경 색상의 영향이 과도하게 반영된다. 이로 인해 FG 분류 성능이 상대적으로 낮다.

3. **모델별 성능**:
   - **Tree 계열**: 색상 정보를 단순 분기로도 충분히 분류 가능 → 100% 성능
   - **SVM**: 비선형 경계로 색상 분리 용이 → 99-100% 성능
   - **KNN**: 거리 기반 분류의 한계 → 상대적 저조

### 6.2 모델별 특성 분석

#### 6.2.1 KNN: 거리 기반 분류의 한계와 극복

**한계**:
- 고차원 공간에서 거리 계산이 비효율적 (Curse of Dimensionality)
- 노이즈에 민감
- 계산 비용이 높음 (O(n) 예측 시간)

**극복 방안**:
1. **PCA 적용**: 2,352차원 → ~300차원으로 축소하여 노이즈 제거 및 속도 향상
2. **Manhattan Distance (p=1)**: Euclidean보다 고차원에서 효과적
3. **Distance Weighting**: 가까운 이웃에 더 높은 가중치 부여

**실험 과정에서의 발견**:

초기에는 PCA 없이 실험했으나, 성능이 92.61%로 저조하였다. PCA를 적용한 후 97.93%로 향상되었으며, 이는 **전처리의 중요성**을 보여준다. 특히 PCA 90%가 최적이었으며, 이는 적절한 차원 축소가 노이즈 제거 효과를 가져온다는 것을 의미한다.

#### 6.2.2 SVM: 비선형 결정 경계의 학습

**강점**:
- RBF 커널을 통해 복잡한 비선형 패턴 학습 가능
- Regularization(C)을 통해 과적합 제어

**튜닝 포인트**:
- **C**: 규제 강도. 높을수록 마진이 좁아지고 과적합 위험 증가
- **gamma**: 커널 폭. 높을수록 결정경계가 복잡해지고 과적합 위험 증가

**실험 과정에서의 발견**:

gamma='scale'이 고차원 데이터에서 비효율적이라는 것을 발견한 후, 명시적인 값(0.001)으로 변경하여 성능을 향상시킬 수 있었다. 또한 C 값을 35.0으로 높여 언더피팅을 완화하였다.

#### 6.2.3 Decision Tree: 해석 가능성과 과적합

**강점**:
- Feature Importance를 통해 의사결정 과정 해석 가능
- 전처리(Scaling) 불필요

**한계**:
- 고차원 공간에서 과적합 쉬움
- 단일 분기로 복잡한 패턴 학습 어려움

**실험 결과**: Digit Task에서 77.13%로 가장 낮은 성능을 기록하였다. 이는 단일 Tree가 복잡한 형태 패턴을 학습하기 어렵기 때문이다.

#### 6.2.4 Random Forest: Bagging의 효과

**강점**:
- 여러 Tree의 평균/투표로 variance 감소
- 과적합에 강함
- Feature Importance 제공

**최적화 포인트**:
- **n_estimators**: 1200이 최적 (이후 포화)
- **min_samples_split/leaf**: 제한 없이(2, 1) 사용하는 것이 효과적

**실험 과정에서의 발견**:

n_estimators를 300에서 1200으로 증가시켜 0.23% 향상을 달성하였다. 시간 대비 효과는 낮지만, 최적 성능을 달성하기 위해 1200을 선택하였다.

#### 6.2.5 XGBoost: Boosting의 효과

**강점**:
- 잔차(residual) 학습을 통해 오분류 샘플의 경계를 점진적으로 보정
- Regularization을 통해 과적합 제어
- Early Stopping 지원

**핵심 특징**:

XGBoost는 이전 모델의 오분류를 다음 모델이 학습하여 **점진적으로 성능을 향상**시킨다. Digit Task에서 가장 효과적인 이유는 복잡한 형태 패턴을 단계적으로 학습할 수 있기 때문이다.

### 6.3 앙상블 실험 결과

#### 6.3.1 Weighted Voting (Soft Voting)

**구현 방법**:
- 각 모델의 Validation F1-score를 가중치로 사용
- Soft Voting (확률 기반)

```python
# F1-score 기반 가중치 계산
model_f1_scores = {'rf': 0.9340, 'xgb': 0.9506, 'knn': 0.9259}
total_f1 = sum(model_f1_scores.values())
weights = [f1 / total_f1 for f1 in model_f1_scores.values()]

voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('knn', knn_model)],
    voting='soft',
    weights=weights
)
```

**결과**: 단일 모델(XGBoost) 대비 소폭 향상 또는 유사한 성능

**분석**: 앙상블의 효과가 제한적인 이유는 **XGBoost가 이미 충분히 강력한 모델**이기 때문이다. 앙상블은 base 모델들의 예측이 서로 complementary할 때 효과적인데, XGBoost가 대부분의 샘플을 정확히 분류하므로 추가적인 향상이 어렵다.

#### 6.3.2 Stacking

**구현 방법**:
- Base 모델: RF, XGBoost, KNN
- Meta 모델: Logistic Regression

**결과**: Voting과 유사한 성능

**분석**: Stacking도 Voting과 마찬가지로 XGBoost가 이미 강력하여 추가적인 향상이 제한적이었다.

---

## 7. 결론 및 향후 연구

### 7.1 연구 결과 요약

본 연구를 통해 다음과 같은 결론을 도출하였다:

1. **색상 분류는 전통 ML로도 충분**: FG/BG Color Task는 모든 모델에서 99-100%의 정확도를 달성하여, 색상 정보가 RGB 공간에서 명확히 분리되어 있음을 확인하였다.

2. **Digit 분류는 모델 선택이 중요**: 형태적 복잡성으로 인해 모델 간 성능 차이가 크게 나타났다. KNN+PCA(97.93%)가 별도 실험에서 최고 성능을 달성했으며, XGBoost(95.27%)가 PCA 없이 최고 성능을 보였다. Decision Tree(77.13%)가 가장 낮았다.

3. **전처리의 중요성**: KNN에 PCA를 적용하면 Digit Task에서 97.93%의 최고 성능을 달성할 수 있었다. 이는 **적절한 전처리가 모델 성능에 미치는 영향**이 크다는 것을 보여준다.

4. **하이퍼파라미터 튜닝의 효과**: SVM의 경우 C와 gamma 튜닝만으로 87.01% → 94.24%로 약 7.2% 성능 향상을 달성하였다. 체계적인 튜닝이 중요하다.

5. **Task별 하이퍼파라미터 최적화**: KNN의 경우 Digit Task에는 Manhattan 거리(p=1)가, 색상 Task(FG/BG)에는 Euclidean 거리(p=2)가 더 효과적임을 확인하였다. 이는 **데이터 특성에 맞는 하이퍼파라미터 선택의 중요성**을 보여준다.

6. **앙상블의 한계**: Weighted Voting, Stacking 등의 앙상블 기법은 이미 강력한 단일 모델(XGBoost)이 있을 때는 효과가 제한적이다.

### 7.2 연구 과정에서 배운 점

본 프로젝트를 통해 머신러닝의 실질적인 경험을 쌓을 수 있었다:

1. **가설 → 실험 → 분석의 반복**: "왜 이 성능이 나왔는가?"라는 질문에 답하기 위해 다양한 실험을 설계하고, 결과를 분석하는 과정을 경험하였다. 예를 들어, SVM의 gamma='scale'이 비효율적이라는 것을 발견한 후, 명시적인 값으로 변경하여 성능을 향상시킬 수 있었다.

2. **하이퍼파라미터의 물리적 의미 이해**: 단순히 GridSearch를 돌리는 것이 아니라, 각 파라미터가 모델의 학습에 어떤 영향을 미치는지 이해하게 되었다 (예: SVM의 C-gamma trade-off, RF의 n_estimators 포화 현상).

3. **전처리의 중요성 체감**: 같은 모델이라도 전처리(Deskewing, PCA)에 따라 성능이 크게 달라진다는 것을 직접 확인하였다. 특히 KNN에 PCA를 적용한 경우 97.93%의 최고 성능을 달성하여, 전처리의 중요성을 실험적으로 입증하였다.

4. **데이터 이해의 중요성**: t-SNE, Confusion Matrix 등을 통해 데이터의 구조를 이해하는 것이 모델 선택과 튜닝에 큰 도움이 된다는 것을 배웠다. 예를 들어, Font 기반 데이터의 분포를 분석하여 최적 비율(10%)을 결정할 수 있었다.

5. **시간 대비 효과의 고려**: n_estimators를 1200으로 증가시켜 0.23% 향상을 달성했지만, 학습 시간이 4배 이상 증가한다. 실무에서는 시간 대비 효과를 고려하여 적절한 절충안을 선택해야 한다.

### 7.3 향후 연구 방향

1. **PCA의 체계적 적용**: 모든 모델에 PCA를 적용하여 성능 향상 여부를 확인 (현재는 KNN에만 적용)

2. **더 다양한 전처리 기법**: HOG, LBP 등 feature engineering 적용

3. **딥러닝과의 비교**: CNN 등 신경망 모델과의 성능 비교 (본 연구 범위 외)

4. **교수님 Test Set 평가**: 실제 Test Set에서의 성능 확인 및 분석

5. **하드 샘플 마이닝**: 오분류 샘플에 집중하여 추가 학습

---

## 참고문헌

[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.

[2] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'16), pp. 785-794, 2016.

[3] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.

[4] C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, no. 3, pp. 273-297, 1995.

[5] scikit-learn Documentation, https://scikit-learn.org/stable/

---

## 부록: 최종 하이퍼파라미터 설정

```yaml
# configs/models.yaml

models:
  knn:
    # Digit Task (PCA 적용)
    digit:
      n_neighbors: 7
      weights: "distance"
      p: 1  # Manhattan distance (형태 분류)
      use_pca: true
      pca_n_components: 0.99
    # FG/BG Task (색상 분류)
    fg:
      n_neighbors: 5
      weights: "distance"
      p: 2  # Euclidean distance (RGB 색상 공간)
      use_pca: false
    bg:
      n_neighbors: 5
      weights: "distance"
      p: 2  # Euclidean distance (RGB 색상 공간)
      use_pca: false
    
  svm:
    # 모든 Task 동일
    digit/fg/bg:
      C: 40.0
      gamma: 0.002
      decision_function_shape: "ovr"
    
  tree:
    # 모든 Task 동일
    digit/fg/bg:
      criterion: "entropy"
      max_depth: null
      min_samples_split: 2
      min_samples_leaf: 1
    
  rf:
    # Digit Task
    digit:
      n_estimators: 1200
      criterion: "entropy"
      max_depth: null
      min_samples_split: 2
      min_samples_leaf: 1
    # FG/BG Task (100% 달성)
    fg/bg:
      n_estimators: 300
      criterion: "entropy"
      max_depth: null
      min_samples_split: 2
      min_samples_leaf: 1
    
  xgb:
    # 모든 Task 동일 (최적화된 하이퍼파라미터)
    digit/fg/bg:
      n_estimators: 500
      learning_rate: 0.19
      max_depth: 5
      gamma: 0.001
      reg_alpha: 0.092
      reg_lambda: 1.49
      subsample: 0.97
      colsample_bytree: 0.99
```

