# 🎨 Colored MNIST Classification — **Colab-first & develop base**

Soongsil Univ. IT융합 · 머신러닝(2024-2)

> **핵심 원칙**  
> 1) **개발/실험은 Colab에서**, 코드는 GitHub로 관리  
> 2) **팀 기준 브랜치 = `develop`**, 개인 작업은 `nb/<topic>`에서  
> 3) **대용량 산출물(모델, 로그, 그림)은 GitHub 금지** → **Drive 전용 폴더**에만 저장

---

## 📌 이 문서로 무엇이 해결되나요?

- Colab에서 **어떻게 레포를 시작/열고/동기화**하는지  
- **commit / push / pull**을 Colab에서 **정확히** 어떻게 하는지  
- `develop`과 `nb/<topic>` **브랜치 협업 규칙**  
- **대용량 파일이 실수로 GitHub에 올라가는 걸** 100% 방지하는 방법  
- 매일 아침 **3줄 루틴**, **체크리스트**, **에러 해결법(FAQ & 트러블슈팅)**

---

## ✅ 공식 개발 환경

- **개발 방식:** Google **Colab** + Google **Drive** + **GitHub**  
- **Python:** Colab 기본(3.10.x)  
- **핵심 라이브러리:** PyTorch · TorchMetrics · TensorBoard · scikit-learn · UMAP · Grad-CAM  
- **브랜치 전략:**  
  - `main`: 최종 결과/발표용 (직접 push 금지)  
  - **`develop`: 팀 기준/통합 브랜치**  
  - `nb/<topic>`: 개인·기능별 작업 브랜치 (예: `nb/eda`, `nb/baseline`, `nb/gradcam`)  
- **저장 정책:** **코드/노트북만 GitHub**, **대용량 산출물은 Drive 전용**  

---

## 🧭 전체 흐름(한눈에 보기)

1. **최초 1회**: Colab에서 Drive 마운트 → 레포 **클론** → **`develop` 전환**  
2. **매 세션(매번)**: Drive 마운트 → 레포 폴더 이동 → **`develop` pull 동기화**  
3. **작업 시작**: `develop`에서 **`nb/<topic>` 분기** → 노트북 실행  
4. **업로드**: **출력 제거** → **`git add` → `git commit` → `git push`** → **PR(base=develop)**  
5. **산출물 방지**: **레포 밖 폴더 저장** 또는 **`.gitignore`** + 추적 해제

---

## 🔐 사전 준비(개인 1회) — GitHub Personal Access Token

1. GitHub → Settings → **Developer settings** → **Personal access tokens** → **Tokens (classic)**  
2. “Generate new token (classic)” → **Note:** `colab-access` / **Scope:** `repo` / **Expiration:** 90 days  
3. 생성된 토큰(예: `ghp_xxx...`) **복사 & 개인 보관** (**절대 공유 금지**)  

> **중요:** Colab에서 최초 `git clone` 시 토큰이 URL에 쓰입니다. 팀원별로 본인 토큰을 사용하세요.

---

## 🚀 1) Colab **최초 세팅**(팀원별 1회만)

> 아래 블록은 **Colab 코드 셀**에 그대로 붙여넣고 실행하세요.  
> Colab에서는 **파이썬 셀** 안에서 `!`로 쉘 명령, `%cd`로 디렉터리 이동을 합니다.

```python
# 1) Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2) 작업 디렉토리 이동
%cd /content/drive/MyDrive

# 3) GitHub에서 레포 클론 (개인 토큰 사용)
USER = "<YOUR_GITHUB_ID>"          # 예: deephoon
TOKEN = "<YOUR_PERSONAL_ACCESS_TOKEN>"  # ghp_로 시작
REPO = "colored-mnist-classification"

!git clone https://{TOKEN}@github.com/{USER}/{REPO}.git
%cd {REPO}

# 4) Git 사용자 정보(1회)
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# 5) 팀 기준 브랜치 develop으로 전환 + 최신화
!git checkout develop
!git pull origin develop

# 6) 상태 확인
!git status
```

> 성공하면 레포는 **`/content/drive/MyDrive/colored-mnist-classification`**에 영구 저장됩니다.  
> 다음부터는 다시 클론하지 않습니다.

---

## 🔁 2) **매 세션 시작** 루틴(매번 동일, 4줄)

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd /content/drive/MyDrive/colored-mnist-classification
!git checkout develop && git pull origin develop
```

- 여기까지 하면 **항상 팀 기준 최신 상태**가 됩니다.  
- 이후 새 작업은 **항상 `develop`에서 `nb/<topic>`로 분기**해서 진행합니다.

---

## 🔀 3) **브랜치 워크플로우**(develop 기반)

### A) 새 작업 시작(새 브랜치 만들기)
```python
!git checkout develop && git pull origin develop
!git checkout -b nb/<topic>   # 예: nb/eda, nb/baseline, nb/gradcam
```

### B) 기존 작업 이어서(이미 있는 브랜치 계속)
```python
!git fetch --all --prune
!git checkout nb/<topic> || git checkout -t origin/nb/<topic>
!git pull
```

### C) 최신 develop을 내 브랜치에 반영(merge)
```python
!git checkout develop && git pull origin develop
!git checkout nb/<topic>
!git merge develop
# 충돌나면 노트북/코드에서 <<<< ==== >>>> 표시 부분 수정 → 아래 커밋/푸시 진행
```

> **Tip:** 브랜치 이름은 **작업 단위**로 작게(예: `nb/fix-aug`, `nb/cm-plot`) 만들면 협업이 깔끔합니다.

---

## 📚 4) **노트북 실행 순서**

1. `00_project_intro.ipynb`  — 환경 점검 / 시드 고정 / 버전 로그  
2. `01_data_build_colored.ipynb` — Colored MNIST 생성  
3. `02_eda.ipynb` — 분포/샘플 시각화  
4. `03_baseline_train.ipynb` — 기본 CNN 학습  
5. `04_improved_train.ipynb` — 정규화/Dropout/BN/증강 비교  
6. `05_evaluation.ipynb` — Accuracy/Confusion Matrix/Grad-CAM  
7. `06_ablation_study.ipynb` — Ablation(요인별 영향)  
8. `07_report_figures.ipynb` — 보고서용 그림 생성

---

## 🤝 5) 협업 규칙(필수)

- **브랜치:**  
  - `main` : **최종 발표/제출만** (직접 push 금지)  
  - `develop` : **모든 작업의 기준/통합**  
  - `nb/<topic>` : **개인 작업/기능 단위**  
- **커밋 Prefix 컨벤션:**  
  - `[nb]` 노트북 / `[fig]` 그림 / `[doc]` 문서 / `[conf]` 환경/설정  
  - 예: `[nb] baseline: acc/loss curves + seed log`  
- **PR 작성 규칙:**  
  - **base = `develop`**, compare = `nb/<topic>`  
  - 본문에 **결과 스크린샷 1장 + 변경점 3줄 요약**  
  - Merge 후 **작업 브랜치 삭제**  
- **커밋 전 필수:**  
  - Colab 메뉴 **런타임 → 모든 출력 지우기** (diff 최소화, 리뷰 용이)  
- **대용량 파일(모델/로그/그림):**  
  - **GitHub 금지** → **Drive 전용 경로에만 저장** (아래 6절)

---

## 💾 6) 산출물 저장 — **GitHub에 안 올라가게** 하는 2가지 방법

### 방법 1) **레포 밖** 전용 폴더에 저장(**가장 안전**)
- **레포(코드/노트북):** `/content/drive/MyDrive/colored-mnist-classification` (**Git 추적 O**)  
- **산출물 전용(권장):** `/content/drive/MyDrive/colored-mnist-results` (**Git 추적 X**)  

노트북에서 **BASE 경로**를 이렇게 씁니다:
```python
BASE_RESULTS = "/content/drive/MyDrive/colored-mnist-results"
# BASE_RESULTS/figures, BASE_RESULTS/logs, BASE_RESULTS/models ... 사용
```
→ 설령 `!git add .`를 해도 **레포 밖**이라 Git이 추적하지 않습니다.

### 방법 2) 레포 **안**을 쓴다면, 반드시 **`.gitignore`**로 차단
레포 내부 `data/`나 `results/`를 써야 한다면 **반드시** `.gitignore`를 설정하세요:

```gitignore
# 결과/데이터 디렉토리
data/
results/

# 모델/체크포인트/바이너리
*.pt
*.pth
*.ckpt
*.bin
*.onnx
*.tflite

# 넘파이/피클/캐시
*.npy
*.npz
*.pkl
*.cache

# 로그/텐서보드
*.log
events.out.tfevents.*
**/logs/**
**/tensorboard/**

# 이미지/문서(정책상 레포 미보관 시)
**/figures/**
*.png
*.jpg
*.jpeg
*.svg
*.pdf

# 주피터 체크포인트/파이캐시
.ipynb_checkpoints/
__pycache__/
```

> **주의:** `.gitignore`는 “앞으로 추가될 파일”만 막습니다.  
> 이미 `git add`로 추적 중이거나 푸시된 파일은 **아래 7절 ‘추적 해제’**가 필요합니다.

---

## 🧹 7) 이미 올라갔거나 추적 중인 **대용량 파일 제거(추적 해제)**

1) **추적만 끊고 파일은 로컬/Drive에 유지**  
```bash
!git rm --cached -r results/ data/
!git rm --cached *.pt *.pth *.ckpt *.onnx *.npy *.npz *.pkl *.log
```

2) `.gitignore` 추가/수정 → 커밋/푸시  
```bash
!git add .gitignore
!git commit -m "[conf] add .gitignore & untrack large artifacts"
!git push
```

이후부터 동일 경로/패턴 파일은 **자동 무시**됩니다.

---

## 🧯 8) “실수 방지” 프리커밋 훅(선택)

- **목표:** 특정 크기(예: **20MB 이상**) 파일이 커밋되지 못하게 **사전 차단**  
- 방법 A: `.git/hooks/pre-commit`에 간단한 크기 검사 스크립트 추가  
- 방법 B: `pre-commit` 프레임워크의 `check-added-large-files` 훅 사용  
- 효과: 초심자 실수로 대용량 산출물 푸시하는 걸 **원천 차단**

---

## ⬆️ 9) Colab에서 **commit / push / pull** 정확 사용법

### A) **커밋 전 출력 제거**(강력 권장)
- Colab 메뉴 **런타임 → 모든 출력 지우기**  
- 또는:
```bash
!jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/<file>.ipynb
```

### B) **커밋(Commit)**
```bash
# 현재 브랜치 확인 (반드시 nb/<topic> 브랜치에서 작업)
!git branch

# 변경 파일 스테이징
!git add notebooks/<file>.ipynb    # 필요 시 정책 허용된 results 경로만 선택적으로 추가

# 메시지 컨벤션 예
!git commit -m "[nb] baseline: acc/loss curves + seed log"
```

### C) **푸시(Push)**
```bash
# 최초 푸시(추적 브랜치 설정 포함)
!git push -u origin HEAD

# 같은 브랜치에서 추가 푸시는 간단히:
!git push
```

### D) **풀(Pull)**
```bash
# 현재 브랜치 최신화
!git pull

# 또는 명시적으로:
!git checkout develop && git pull origin develop
```

### E) **충돌(conflict) 해결**
1. `!git status`로 충돌 파일 확인  
2. 충돌 파일 열고 **`<<<<` / `====` / `>>>>`** 구간을 수동 수정  
3. 저장 후:
```bash
!git add .
!git commit -m "[nb] resolve merge conflicts"
!git push
```

### F) **되돌리기(안전 스위치)** — 신중히 사용
```bash
# 스테이징 취소
!git restore --staged <file>

# 파일 내용을 마지막 커밋으로 되돌림
!git restore <file>

# 로컬 변경 전부 취소(주의)
!git reset --hard HEAD

# 원격 기준으로 맞춤(주의)
!git fetch --all && git reset --hard origin/<branch>
```

---

## 📂 10) **폴더 구조(권장)** — Git 레포 vs 결과 전용

```text
/content/drive/MyDrive/
  ├─ colored-mnist-classification/     # Git 레포 (코드/노트북만)
  │   ├─ notebooks/
  │   ├─ configs/
  │   ├─ docs/
  │   └─ README.md
  └─ colored-mnist-results/            # 산출물 전용 (Git 추적 X)
      ├─ data/
      ├─ figures/
      ├─ logs/
      └─ models/
```

> **정책:** 위처럼 **분리**가 가장 안전합니다. 레포 안 `results/`를 쓰면 **.gitignore** 필수.

---

## 🧪 11) 노트북 실행 **체크리스트**

- `00_project_intro.ipynb`에서 **시드 고정** 및 **torch/sklearn 버전, GPU 타입(T4/L4/A100) 로그** 남김  
- 결과 저장 경로가 **Drive 전용 폴더**인지 확인(레포 밖 권장)  
- 커밋 전 **출력 제거** 완료  
- 커밋 브랜치가 **`nb/<topic>`**인지 확인(절대 `develop`/`main`에서 직접 작업 금지)  
- PR은 **base=`develop`**, 본문에 스샷 + 변경점 3줄

---

## ❓ 12) FAQ (초심자용)

**Q1. Colab 새로 켤 때마다 뭘 하죠?**  
A. “**Drive 마운트 → 레포 폴더 이동 → `develop` pull**” 4줄만 실행하면 됩니다. 다시 클론하지 않습니다.

**Q2. `fatal: not a git repository` 에러가 떠요.**  
A. 레포 폴더가 아닙니다. `%cd /content/drive/MyDrive/colored-mnist-classification` 후 `!git status`를 확인하세요.

**Q3. `pathspec 'nb/...' did not match`가 나오는데요?**  
A. 로컬에 브랜치가 없어서 그래요. `!git fetch --all --prune` 후 `!git checkout -t origin/nb/<topic>`로 가져오세요.

**Q4. 실수로 대용량 파일을 올렸어요.**  
A. 7절의 **추적 해제** 절차(`git rm --cached ...`)를 따라 정리 후 `.gitignore`를 커밋하세요.

**Q5. 왜 출력 제거를 해야 하나요?**  
A. 노트북 출력이 포함되면 diff가 커지고 리뷰가 어렵습니다. 항상 “모든 출력 지우기” 후 커밋하세요.

---

## ⚡ 13) **세션 부트업 1셀**(맨 위에 두고 매번 실행)

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd /content/drive/MyDrive/colored-mnist-classification
!git checkout develop
!git pull origin develop
!git status
print("✅ Ready on 'develop'")
```

---

## 📦 14) 제출용 **환경 스냅샷**(재현성 기록)

```bash
!pip freeze > requirements.lock.txt
```

> 팀 제출/재현성 확인을 위해 PR/리포트 직전에 위 파일을 생성/업데이트하세요.

---

## 🧷 15) 핵심 3줄 요약

1) **세션 시작:** `mount → cd → checkout develop → pull`  
2) **작업 브랜치:** `develop`에서 분기한 **`nb/<topic>`**에서만 작업/커밋  
3) **산출물 방지:** **레포 밖 저장** + **.gitignore** (+ 선택: **프리커밋 훅**)

---
