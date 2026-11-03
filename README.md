# DLinear 재현 (VS Code + Google Colab)

논문 **Are Transformers Effective for Time Series Forecasting?** (Zeng et al., AAAI 2023)의 DLinear 결과(Table 2, 다변량)를
공식 코드 [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)로 재현한 기록.

- 논문: <https://arxiv.org/abs/2205.13504>
- 실행 환경: VS Code + Google Colab 확장(`google.colab`), Colab T4 GPU
- 공식 코드는 이 레포에 복사하지 않고, 노트북이 실행 시점에 clone한다.

![DLinear 구조](figures/05_dlinear_architecture.png)

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `DLinear_reproduce.ipynb` | 재현 노트북 (코드 받기 → 패치 → 데이터 → 실험 → 결과 비교 → 시각화) |
| `.vscode/` | 권장 확장(Colab, Jupyter)과 노트북 설정 |
| `scripts/dlinear_check.py` | 공식 설정을 그대로 옮긴 독립 DLinear 재구현 (ETTh1, CPU로도 실행 가능) |
| `scripts/make_figs.py` | 설명용 그림 14장 생성 |
| `scripts/weights_fig.py` | 학습된 DLinear 가중치 시각화 |
| `figures/` | 생성된 그림 |
| `results/` | 재현 결과 (`result.txt`, 비교표) |

## 실행 방법

### VS Code (권장)
1. 레포를 clone해서 폴더를 VS Code로 연다 → "권장 확장 설치" 알림에서 **설치** (Colab + Jupyter).
2. `DLinear_reproduce.ipynb` 열기 → 오른쪽 위 **Select Kernel → Colab → New Colab Server → T4 → Standard**.
   - 처음엔 구글 로그인이 필요하다. 로그인 대기 시간이 1분이라, 브라우저에서 바로 계정 선택 → 허용.
3. 위에서부터 셀 실행.

### 브라우저 Colab
노트북을 Colab에 업로드 → 런타임 유형 GPU → 위에서부터 실행.

## 최신 환경에서 필요한 패치

공식 `requirements.txt`는 `torch==1.9` / Python 3.6 기준이라 **설치하지 않는다** (Colab 기본 환경 사용).
NumPy 2.x에서는 `np.Inf`가 삭제돼 학습 시작 직후 에러가 나므로 한 줄 고친다.

```bash
sed -i 's/np\.Inf/np.inf/' utils/tools.py
```

## 결과

ETTh1, L=336 → T=96 (seed 2021, Colab T4):

| | MSE | MAE |
|---|---|---|
| 재현 | 0.3841 | 0.4047 |
| 논문 Table 2 | 0.375 | 0.399 |

전체 데이터셋 결과는 [`results/`](results/) 참고.

### 주의: 시드 편차
학습률이 매 epoch 절반으로 줄고 patience 3이라 몇 epoch 안에 early stop되므로, 작은 데이터셋(ETTh1 등)은 시드에 따라 결과가 꽤 흔들린다.
독립 재구현(`scripts/dlinear_check.py`)으로 확인한 ETTh1 MSE:

| horizon | 논문 | seed 2021 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|
| T=96 | 0.375 | 0.384 | 0.372 | 0.375 | – |
| T=336 | 0.439 | 0.508 | 0.442 | 0.447 | 0.458 |

한 번 결과로 판단하지 말고 `--itr 3` 평균으로 볼 것.

## 그림 다시 만들기

```bash
pip install numpy pandas matplotlib torch
python scripts/make_figs.py path/to/ETTh1.csv figures
python scripts/dlinear_check.py path/to/ETTh1.csv 96 1 viz   # w_seasonal.npy, w_trend.npy 저장
python scripts/weights_fig.py . figures
```

한글 폰트(Noto Sans CJK KR)가 설치돼 있어야 그림 속 한글이 제대로 나온다.

## 출처
- 공식 코드: [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) (Apache-2.0)
- 데이터: ETT는 [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), 나머지는 Autoformer 저자 제공 구글 드라이브
