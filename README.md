# DLinear · NLinear Colab 재현

논문 **Are Transformers Effective for Time Series Forecasting?** (Zeng, Chen, Zhang, Xu, AAAI 2023)의 선형 모델 결과를 공식 코드 [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)로 Google Colab에서 재현하고, 논문값과 다른 칸의 원인을 추적한 기록입니다.

- 논문: <https://arxiv.org/abs/2205.13504>
- 공식 코드는 이 저장소에 복사하지 않았습니다. 노트북이 실행할 때 내려받고, 필요한 커밋(`04ffd8d`, `7f7a25b`)을 따로 꺼내 씁니다.
- 발표 자료와 대본(한국어, 영어)은 [`docs/`](docs/)에 있습니다.

![DLinear Reproduction](assets/social-preview.png)

[English summary](#english-summary)

## 결론

**1. 재현한 92칸 중 89칸은 논문과 맞거나, 다른 이유를 확인했습니다.**

| 표 | 칸 수 | 결과 |
|---|---|---|
| 다변량 DLinear (Table 2) | 36 | 35칸이 소수 셋째 자리까지 일치. ETTm2 96만 MSE 0.0007 차이 |
| 단변량 DLinear (Table 9) | 16 | 모두 설명됨. 11칸은 스크립트 그대로, 4칸은 학습률, 1칸은 논문 표의 오타 |
| 다변량 NLinear (Table 2) | 20 | 18칸 일치. ETTm2 336(0.0008 차이), 환율 720(0.943 대 1.033)은 원인 모름 |
| 다변량 Linear (Table 2) | 20 | 20칸 모두 일치 |

**2. 차이의 주된 원인은 공식 코드가 논문 이후에 바뀐 것입니다.** 논문 숫자는 2022년 6월 코드(`04ffd8d`)에서 나왔습니다. 그 뒤 가중치 초기화가 빠지고, 예측에 쓰이지 않는 층이 지워지고(그 층이 난수를 소비해서 데이터 섞는 순서가 바뀜), 스크립트 학습률과 테스트 채점 방식이 바뀌었습니다. 자세한 내용: [`confirmed_causes.md`](results/cause_investigation/confirmed_causes.md), 남은 세 칸: [`open_problems.md`](results/cause_investigation/open_problems.md)

**3. 흔들리는 칸이 있습니다.** ETTh2 720, 환율 720 같은 칸은 조기 종료가 학습 초반의 모델을 고른 값이라 시드에 크게 흔들립니다(시드 20개에서 환율 720은 0.49~1.05).

**4. 논문의 분석 실험도 직접 돌렸습니다** (Colab T4, 269회 학습). 결과: [`results/paper_experiments/`](results/paper_experiments/)

| 논문 | 결론 | 재현 |
|---|---|---|
| Table 2 | DLinear가 FEDformer보다 낫다. 환율은 마지막 값 반복이 FEDformer보다 낫다 | 재현됨 |
| Table 5 | (환율) Transformer는 입력 순서를 섞어도 그대로다 | 재현됨. ETTh1에서는 FEDformer·Autoformer도 나빠진다(논문에도 있음) |
| Table 6 | 위치 정보를 빼도 FEDformer·Autoformer는 그대로다 | 재현됨 |
| Table 8 | DLinear가 훨씬 가볍고 빠르다 | 재현됨 |
| Table 3 | 먼 입력을 줘도 성능이 거의 같다 | FEDformer는 숫자까지, Autoformer는 방향만 |
| Figure 4 | 선형 모델만 길게 볼수록 좋아진다 | 재현됨 (전력, 교통, 7개 모델) |
| Table 4 | Informer를 단순하게 만들수록 좋아진다 | 일부. Linear 8칸은 일치, 환율의 중간 단계는 다름 |
| Table 7 | 학습 데이터를 줄여도 나빠지지 않는다 | 일부. FEDformer만 같고 Autoformer는 나빠짐 |

Table 3, Table 7과 Table 4의 가운데 두 모델은 저자 코드가 공개되지 않아 논문 설명대로 구현했습니다.

**5. 개선 가설 1단계(H1~H3)를 실험했습니다.** 가설: [`docs/improvement_hypotheses.md`](docs/improvement_hypotheses.md), 1단계 결과(데이터 9개): [`results/hypotheses/stage1.md`](results/hypotheses/stage1.md). 학습 대신 수식(릿지 회귀)으로 푼 해가 36칸 중 29칸에서 가장 낮았습니다. 마지막 값 빼기는 높이가 움직이는 데이터(ETTh2, ETTm2, 독감)에서만 효과가 있고, DLinear에 더하면 NLinear와 같은 수준이 됩니다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `DLinear_reproduce.ipynb` | 첫 재현. 지금 공식 코드로 Table 2의 DLinear 36칸 |
| `DLinear_extra_experiments.ipynb` | 추가 실험: 시드 편차, 단변량(Table 9), 가중치 그림, `--individual`, 입력 길이 |
| `DLinear_followup_experiments.ipynb` | 논문 당시 코드(`04ffd8d`)로 재실행, Table 8 시간·메모리 |
| `DLinear_traffic_ep30.ipynb` | 교통: 최대 30 epoch로 학습하면 논문값이 나오는지 |
| `DLinear_shuffle_transformers.ipynb` | Table 5 순서 섞기, Transformer 쪽 |
| `DLinear_last_cells_gpu.ipynb` | 원인을 못 찾은 칸을 GPU에서 다시 돌리기 |
| `DLinear_paper_experiments.ipynb` | 논문 분석 실험(Table 2·3·4·6·7·8, Figure 4) |
| `DLinear_hypotheses.ipynb` | 개선 가설 실험 (독감, 날씨, 전력, 교통) |
| `scripts/` | 노트북이 쓰는 실행 스크립트와 그림 스크립트 |
| `results/` | 결과 CSV와 정리 문서. `cause_investigation/`(원인 추적), `paper_experiments/`, `hypotheses/`, `synthetic/`(직접 만든 데이터 실험) |
| `docs/` | 발표 자료와 대본 PDF(한국어, 영어), 개선 가설 |
| `figures/` | 설명용 그림 |

## 실행 방법

1. 노트북을 Colab에서 엽니다(브라우저 Colab, 또는 VS Code의 Colab 확장). 런타임은 GPU(T4)입니다.
2. 위에서부터 실행합니다. 공식 코드와 데이터(저자 제공 Google Drive)는 노트북이 내려받습니다.
3. 전력과 교통은 메모리를 많이 써서 High-RAM 런타임이 필요합니다.

공식 `requirements.txt`(PyTorch 1.9, Python 3.6)는 설치하지 않고 Colab 기본 환경을 씁니다. NumPy 2에서는 `np.Inf`가 없어져서 한 줄만 고칩니다(`sed -i 's/np\.Inf/np.inf/' utils/tools.py`). 그 밖의 공식 코드는 고치지 않습니다.

## 처음 재현했을 때의 기록

지금 공식 코드로 처음 돌렸을 때(2026-11-04)는 36칸 중 3칸만 정확히 맞았습니다. 그때의 표와 추가 실험은 [`results/README.md`](results/README.md)와 [`results/extra_experiments.md`](results/extra_experiments.md)에 그대로 남겨 두었습니다. 원인은 위 "결론"의 2번입니다.

## 출처

- 공식 코드: [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) (Apache-2.0). 발표 자료 27쪽의 그림은 이 저장소의 `pics/DLinear/LookBackWindow.png`입니다.
- 데이터: ETT는 [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), 나머지는 Autoformer 저자가 제공한 Google Drive. 이 저장소에는 데이터를 넣지 않았습니다.

---

## English summary

A Colab reproduction of the linear models (Linear, NLinear, DLinear) from *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023), using the official code, with the cause of every mismatch traced.

- **92 cells reproduced, 89 match the paper or have a confirmed cause.** Multivariate DLinear: 35 of 36 cells match to the third decimal. Linear: 20 of 20. NLinear: 18 of 20. Univariate DLinear: all 16 explained (one is a typo in the paper's table).
- **Main cause of the gaps: the official code changed after the paper.** The paper's numbers come from commit `04ffd8d` (June 2022). Later commits removed the weight initialization, deleted an unused layer (which changed the random number stream and so the data order), and changed learning rates and test scoring.
- **Some cells are unstable.** Where early stopping picks an early epoch (e.g. ETTh2 720, Exchange 720), the result depends strongly on the seed.
- **The paper's analysis experiments were rerun** (269 trainings). Accuracy, shuffling (on Exchange), position information, efficiency and look-back length reproduce. The data-size claim reproduces only for FEDformer.
- **Improvement hypotheses, stage 1 (H1 to H3).** On all 9 datasets, a closed-form ridge solution is the lowest of the methods compared in 29 of 36 cells. Subtracting the last value helps only where the level moves (ETTh2, ETTm2, ILI).

Slides and scripts (Korean and English) are in [`docs/`](docs/). The detailed write-ups are in Korean.
