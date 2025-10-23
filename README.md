# 🎨 Colored MNIST Classification (Jupyter-first, Python 3.10.13)

Soongsil Univ. IT융합 · 머신러닝(2024-2)

## ✅ 공식 개발 환경
- **Python: 3.10.13 (고정)**
- 핵심: PyTorch, TorchMetrics, TensorBoard, scikit-learn, UMAP, Grad-CAM
- Jupyter diff: nbdime (선택: jupytext, nbstripout)

## 1) 가상환경 생성 및 패키지 설치
### A. Conda (권장)
```bash
conda create -n colored-mnist-py310 python=3.10.13 -y
conda activate colored-mnist-py310
conda env update -n colored-mnist-py310 -f environment.yml --prune
# 누락 최소화를 위해 한 번 더(보수적)
pip install -r requirements.txt
```

### B. venv (pip)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) 주피터 커널 등록
```bash
python -m ipykernel install --user --name colored-mnist-py310 --display-name "Python 3.10 (colored-mnist)"
```

## 3) 실행 순서(노트북)
1. `00_project_intro.ipynb`  → 환경 점검/역할/시드 고정
2. `01_data_build_colored.ipynb`
3. `02_eda.ipynb`
4. `03_baseline_train.ipynb`
5. `04_improved_train.ipynb`
6. `05_evaluation.ipynb`
7. `06_ablation_study.ipynb`
8. `07_report_figures.ipynb`

## 4) 협업 규칙(요약)
- 브랜치: `main` / `develop` / `nb/<주제>`
- 커밋 prefix: `[nb]`(노트북), `[fig]`(그림), `[doc]`(문서), `[conf]`(환경/설정)
- PR 본문: 결과 스크린샷 1장 + 변경점 3줄
- 저장 전: **Cell → Clear All Outputs**

## 5) 환경 스냅샷(제출 직전)
- pip 팀: `pip freeze > requirements.lock.txt`
- conda 팀: `conda env export --no-builds > environment.lock.yml`
