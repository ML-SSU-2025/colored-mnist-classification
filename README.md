# 🎨 Colored MNIST Classification (Colab-first)

Soongsil Univ. IT융합 · 머신러닝(2024-2)

---

## ✅ 공식 개발 환경  
- **기반:** Google Colab + Google Drive + GitHub  
- **Python:** Colab 기본 (3.10.x)  
- **핵심:** PyTorch, TorchMetrics, TensorBoard, scikit-learn, UMAP, Grad-CAM  
- **결과 저장:** `/content/drive/MyDrive/colored-mnist-classification/`  
- **브랜치 관리:** `main` / `develop` / `nb/<topic>`  

---

## 1) Colab 최초 세팅 (팀원별 1회만)

```bash
# 1️⃣ Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2️⃣ Drive 경로 이동
%cd /content/drive/MyDrive

# 3️⃣ GitHub에서 클론 (토큰 사용)
USER="<YOUR_GITHUB_ID>"
TOKEN="<YOUR_PERSONAL_ACCESS_TOKEN>"
REPO="colored-mnist-classification"

!git clone https://{TOKEN}@github.com/{USER}/{REPO}.git
%cd {REPO}

# 4️⃣ Git 사용자 설정 (최초 1회)
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# 5️⃣ 상태 확인
!git status
```

✅ 위 과정 후 Drive 내 경로:
```
/content/drive/MyDrive/colored-mnist-classification/
```

---

## 2) 다음 세션(매번 실행)

```bash
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/colored-mnist-classification
!git pull origin main
```

- 새 세션마다 위 3줄만 실행하면 최신 상태로 연결됨  
- `main` 대신 `develop` 또는 `nb/<topic>` 사용 시, 브랜치 워크플로우 참고  

---

## 3) 브랜치 워크플로우 (main 외 브랜치 포함)

### A. 원하는 브랜치로 전환
```bash
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/colored-mnist-classification

!git fetch --all --prune
BRANCH="develop"  # ← 작업 브랜치명 (예: nb/eda, nb/baseline-train)
!git checkout {BRANCH} || git checkout -t origin/{BRANCH}
!git pull
```

### B. 새 실험 브랜치 생성 (develop 기준)
```bash
!git checkout develop || git checkout -t origin/develop
!git pull
!git checkout -b nb/<topic>
```

### C. 기존 브랜치 이어서 작업
```bash
!git fetch --all --prune
!git checkout nb/<topic> || git checkout -t origin/nb/<topic>
!git pull
```

### D. 최신 develop 반영
```bash
!git checkout develop
!git pull
!git checkout nb/<topic>
!git merge develop
```

---

## 4) 실행 순서(노트북)

1. `00_project_intro.ipynb`  → 환경 점검 / 시드 고정  
2. `01_data_build_colored.ipynb`  
3. `02_eda.ipynb`  
4. `03_baseline_train.ipynb`  
5. `04_improved_train.ipynb`  
6. `05_evaluation.ipynb`  
7. `06_ablation_study.ipynb`  
8. `07_report_figures.ipynb`  

---

## 5) 협업 규칙(요약)

- **브랜치:** `main` / `develop` / `nb/<topic>`  
- **커밋 prefix:** `[nb]` 노트북 / `[fig]` 그림 / `[doc]` 문서 / `[conf]` 설정  
- **PR 본문:** 결과 스크린샷 1장 + 변경점 3줄 요약  
- **커밋 전:** 반드시 “Cell → 모든 출력 지우기” 실행  
- **대용량 파일:** (모델, 로그 등) → Drive 저장만 허용  

---

## 6) 결과 저장 구조

```
/MyDrive/colored-mnist-classification/
  ├─ data/
  ├─ notebooks/
  ├─ results/
  │   ├─ figures/
  │   ├─ logs/
  │   └─ reports/
```

> 실험 결과는 Drive 내에 자동 저장되도록 설정 (예: `results/figures/`)

---

## 7) 변경사항 업로드 (PR 생성 절차)

```bash
# 출력 정리
!jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/<file>.ipynb

# 변경 반영
!git add notebooks/<file>.ipynb results/*
!git commit -m "[nb] <변경 내용 요약>"

# 푸시 (작업 브랜치 유지)
!git push -u origin HEAD
```

> GitHub에서 Pull Request 생성  
> base 브랜치: `develop` (main 직접 푸시 금지)

---

## 8) 환경 스냅샷(제출 직전)

- Colab 세션 패키지 버전 고정:  
  `pip freeze > requirements.lock.txt`

---

## 9) 문제 해결 가이드

| 증상 | 원인 | 해결 |
|------|------|------|
| fatal: not a git repository | 경로 잘못됨 | %cd /content/drive/MyDrive/colored-mnist-classification |
| pathspec ‘nb/…’ did not match | 로컬 브랜치 없음 | !git fetch --all --prune 후 checkout |
| 403 에러 | PAT 만료 | 새 토큰 발급 후 URL 갱신 |
| diff 너무 큼 | 출력 미삭제 | 모든 출력 지우기 후 커밋 |
| 커밋 이름 오류 | 사용자 정보 누락 | git config 재설정 |

---

## 10) 핵심 요약

| 단계 | 명령 | 설명 |
|------|------|------|
| 1️⃣ | Drive + Clone (1회) | mount → cd → clone |
| 2️⃣ | 세션 시작 | mount → cd → git pull |
| 3️⃣ | 브랜치 전환 | checkout nb/<topic> |
| 4️⃣ | 작업 후 업로드 | add → commit → push |
| 5️⃣ | PR 생성 | GitHub에서 리뷰 후 Merge |

---

## 11) 자동 세팅 셀 (복붙용)

```bash
USER="<YOUR_GITHUB_ID>"
REPO="colored-mnist-classification"
TOKEN="<YOUR_PAT>"
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, subprocess
BASE="/content/drive/MyDrive"
REPO_DIR=f"{BASE}/{REPO}"

def run(cmd): print("$", cmd); subprocess.run(cmd, shell=True, check=False)

if not os.path.isdir(REPO_DIR):
  os.chdir(BASE)
  run(f"git clone https://{TOKEN}@github.com/{USER}/{REPO}.git")
else:
  os.chdir(REPO_DIR)
  run("git pull origin main")

print("✅ Ready:", REPO_DIR)
```

---

## ✅ 최종 메모

- Colab에서 동일 환경으로 실행 가능  
- 각자 브랜치로 실험 후 develop 병합  
- main은 오직 발표용 / 최종 결과만 반영  
- 노트북은 반드시 출력 제거 후 커밋  
