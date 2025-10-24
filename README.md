# 🎨 Colored MNIST Classification (Colab-first, develop base)

Soongsil Univ. IT융합 · 머신러닝(2025-2)

---

## ✅ 공식 개발 환경
- **개발 방식:** Google Colab + Google Drive + GitHub  
- **Python:** Colab 기본(3.10.x)  
- **핵심 라이브러리:** PyTorch · TorchMetrics · TensorBoard · scikit-learn · UMAP · Grad-CAM  
- **브랜치 전략:** `main`(최종) · **`develop`(팀 기준/통합)** · `nb/<topic>`(개인/기능별 작업)  
- **저장 원칙:** **코드·노트북만 GitHub**. **대용량 산출물(모델, 로그, 그림)은 Drive**에만 저장  

---

## 🧭 전체 흐름 (한눈에 보기)
1) **최초 1회**: Colab에서 **Drive 마운트 → 레포 클론 → `develop` 전환**  
2) **매 세션**: **Drive 마운트 → 레포 폴더 이동 → `develop` pull**  
3) **작업**: `develop`에서 **새 브랜치 `nb/<topic>` 분기**, 노트북 실행  
4) **업로드**: 출력 제거 → **add/commit/push** → **PR(base = develop)**  
5) **산출물**: **Drive 전용 폴더** 또는 **`.gitignore`**로 **Git 차단**  

---

## 0) GitHub 보안(PAT) 및 Colab 권장 설정
- **개인별 PAT(토큰)**을 발급해 사용합니다. *(Settings → Developer settings → Tokens(classic) → repo 권한)*  
- 토큰은 **절대 공유 금지**.  
- **토큰 관리 팁**  
  - A안: Drive에 **영구 클론** 후, 이후에는 `git pull/push`만 사용(토큰 재입력 최소화)  
  - B안: Colab **User secrets** 기능으로 토큰 저장(`github_token`) 후 코드에서 불러오기  

---

## 1) Colab 최초 세팅 (팀원별 1회만)
1. **Drive 마운트**  
2. **`/content/drive/MyDrive`로 이동**  
3. **레포 클론**: `https://<TOKEN>@github.com/<USER_OR_ORG>/colored-mnist-classification.git`  
4. **Git 사용자 정보 등록**: 이메일/이름  
5. **팀 기준 브랜치 전환**: `develop` 체크아웃 + `git pull origin develop`  
6. **상태 확인**: `git status`  

> 완료 후 프로젝트 경로: **`/content/drive/MyDrive/colored-mnist-classification/`**

---

## 2) 세션 시작 루틴 (매번 동일 · 4줄)
1) Drive 마운트  
2) 프로젝트 폴더로 이동  
3) `develop` 체크아웃  
4) `git pull origin develop`  

> 이렇게 하면 매 세션 동일한 기준 상태에서 출발합니다.

---

## 3) 브랜치 워크플로우 (develop 기반 협업)
- **새 작업 시작**:  
  1) `git checkout develop && git pull origin develop`  
  2) `git checkout -b nb/<topic>`  *(예: `nb/eda`, `nb/baseline`, `nb/gradcam`)*

- **기존 작업 이어서**:  
  1) `git fetch --all --prune`  
  2) `git checkout nb/<topic> || git checkout -t origin/nb/<topic>`  
  3) `git pull`

- **최신 develop 반영**(내 작업 브랜치로 머지):  
  1) `git checkout develop && git pull origin develop`  
  2) `git checkout nb/<topic>`  
  3) `git merge develop`  
  4) 충돌 시 파일 열어 머커(`<<<< ==== >>>>`) 정리 → `git add . && git commit`

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

## 5) 협업 규칙(필수)
- **브랜치**: `main`(최종) / `develop`(통합) / `nb/<topic>`(작업)  
- **커밋 prefix**: `[nb]` 노트북 · `[fig]` 그림 · `[doc]` 문서 · `[conf]` 환경  
- **PR 규칙**: base = `develop`, 본문에 **결과 스크린샷 1장 + 변경점 3줄**  
- **커밋 전 필수**: Colab **런타임 → 모든 출력 지우기** (diff 최소화)  
- **대용량 파일**: GitHub 금지 → **Drive에만 저장**(아래 6, 7절 참고)

---

## 6) 산출물 저장 정책 (두 가지 중 “1+2 조합” 권장)
### 6-1) **레포 밖 경로**에 저장(가장 안전)
- 레포: `/content/drive/MyDrive/colored-mnist-classification` (**Git 추적 O**)  
- **산출물 전용**: `/content/drive/MyDrive/colored-mnist-results` (**Git 추적 X**)  
- 노트북에서 결과 경로를 **전용 폴더**로 고정(예: `BASE_RESULTS=/content/drive/MyDrive/colored-mnist-results`)

→ **실수로 `git add .` 해도 레포 밖 파일은 추적되지 않습니다.**

### 6-2) 레포 내부를 쓰면 **반드시 `.gitignore`로 차단**
(중략 — 동일 내용 유지)
