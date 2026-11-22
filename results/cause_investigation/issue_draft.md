# 저자 문의용 GitHub issue 초안

올릴 곳: https://github.com/cure-lab/LTSF-Linear/issues (New issue). 아직 올리지 않았다.

올리기 전에 확인한 기존 issue (2026-11-13 기준)
- #107 (열림, 답변 없음): NLinear의 ILI·Traffic 결과가 재현되지 않는다는 질문. 같은 종류지만 칸이 다르다.
- #109 (열림): Table 2·9의 Transformer 하이퍼파라미터를 묻는 질문. 선형 모델 칸은 아니다.
- #27 (닫힘): 단변량 ETTh1 Linear 96 값이 이상하다는 질문. 메인테이너가 "세 선형 모델을 같은 학습률과 배치로 돌려서 Linear가 덜 학습된 것 같다, 표를 고치겠다"고 답했다.

2026-11-22에 고친 것: 단변량 ETTh2 96은 해결되어 질문에서 뺐다. NLinear ETTm2 336의 "배치 8" 후보는 이웃 칸 검증에서 탈락해 지웠다. 환율 720에 시험한 것을 최신으로 채웠다. 확인 질문 두 개(단변량 학습률, Table 8의 변수 수)를 끝에 붙였다.

아래는 그대로 붙여 넣을 본문이다.

---

**Title:** Three cells I could not reproduce: NLinear Exchange-720, NLinear ETTm2-336, DLinear ETTm2-96 (Table 2)

Hi, thank you for releasing the code. I reproduced the LTSF-Linear tables (Table 2 for DLinear / NLinear / Linear and Table 9 for DLinear, 92 cells in total) and could match 89 of them to three decimals in both MSE and MAE, using the code at the commit of the time (04ffd8d for DLinear, 7f7a25b for NLinear / Linear) with seed 2021. Three cells remain, and I would be grateful for any hint about how they were produced.

**1. NLinear, Exchange, horizon 720 (Table 2: 1.033 / 0.780)**

At commit 7f7a25b with `scripts/EXP-LongForecasting/Linear/exchange_rate.sh` and `model_name=NLinear`, I get 0.9430 / 0.7283. The other three Exchange horizons match with the same script (0.089 / 0.208, 0.180 / 0.300, 0.331 / 0.415). For 720 I also tried:

- learning rates 0.05 to 0.0001 with batch 32, and three learning rates with batch 8 and 16: MSE 0.883 to 1.123, MAE 0.703 to 0.796
- seeds 2021 to 2025: 0.943 to 0.958 / 0.728 to 0.734
- look-back 96, 192, 512, 720: 0.900 to 1.015 / 0.716 to 0.752
- patience, epoch limit, `lradj`, `--itr 3` (each iteration), `--use_amp`, `--individual` (commit b9ca453), the current main branch, CPU and a T4 GPU
- ten variants of the normalisation (window mean, first value, median, mean of the last 5 or 25 steps, dividing by the window std, no bias, and others): 0.869 to 0.984 / 0.696 to 0.746, and every variant breaks at least one of the other three horizons

None gives 1.033 and 0.780 together. In all my runs an MSE near 1.03 comes with an MAE of 0.73 to 0.76, so 0.780 looks like it came from a different kind of run. Could you share the setting or code version for this cell?

**2. NLinear, ETTm2, horizon 336 (Table 2: 0.274 / 0.327)**

At commit 7f7a25b with `scripts/EXP-LongForecasting/Linear/ettm2.sh` (lr 0.01, batch 32) I get 0.2748 / 0.3271, so MSE rounds to 0.275. The other three horizons match with the script (720 with the lr 0.01 of commit 726695f). Batch 8 gives 0.2741 for this cell, but it breaks the other three horizons, so I do not think that is the answer.

**3. DLinear, ETTm2, horizon 96 (Table 2: 0.167 / 0.260)**

At commit 04ffd8d with `scripts/EXP-LongForecasting/DLinear/ettm2.sh` (lr 0.001, batch 32) I get 0.1677 / 0.2598, so MSE rounds to 0.168. The other 35 DLinear cells of Table 2 match. This cell stops early (best epoch 4) and moves between 0.166 and 0.171 across seeds, so the gap may simply be noise, but with seed 2021 I always get 0.1677 on CPU, on a T4, and with `--use_amp`.

Cells 2 and 3 have the same pattern: both are ETTm2, MAE matches, and MSE is 0.0007 to 0.0008 too high.

**Two things I would like to confirm**

- Table 9 (univariate DLinear): with commit 04ffd8d, ETTh2 96 and 192 match the paper with lr 0.05 (the script has 0.005), and ETTm2 336 and 720 match with lr 0.001 (the script has 0.01). Is that how they were run? Also, the DLinear ETTh1-720 entry (0.189 / 0.359) is identical to the Linear ETTh1-96 entry, while the June 2022 README has 0.119 / 0.274, which is what I get.
- Table 8: the parameter counts of the Transformer models match what I get with 862 input variables (Traffic), for example 13,605,214 for Transformer, not with the 321 variables of Electricity (11,665,729). Is the table's dataset label correct?

A side note: in `LTSF-Benchmark.md`, second table, the DLinear and NLinear row labels look swapped for Electricity, Exchange, Traffic, Weather and ILI compared with Table 2 of the paper.

Thank you.
