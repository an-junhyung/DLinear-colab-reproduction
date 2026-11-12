# 저자 문의용 GitHub issue 초안

올릴 곳: https://github.com/cure-lab/LTSF-Linear/issues (New issue)

올리기 전에 확인한 기존 issue (2026-11-13 기준)
- #107 (열림, 답변 없음): NLinear의 ILI·Traffic 결과가 재현되지 않는다는 질문. 같은 종류지만 칸이 다르다.
- #109 (열림): Table 2·9의 Transformer 하이퍼파라미터를 묻는 질문. 선형 모델 칸은 아니다.
- #27 (닫힘): 단변량 ETTh1 Linear 96 값이 이상하다는 질문. 메인테이너가 "세 선형 모델을 같은 학습률과 배치로 돌려서 Linear가 덜 학습된 것 같다, 표를 고치겠다"고 답했다.

아래는 그대로 붙여 넣을 본문이다.

---

**Title:** Settings for two cells: univariate ETTh2-96 (Table 9) and NLinear Exchange-720 (Table 2)

Hi, thank you for releasing the code. I reproduced the LTSF-Linear tables on Colab and was able to match almost every cell to three decimals by using the code at the commit of the time (04ffd8d for DLinear, 7f7a25b for NLinear/Linear) with seed 2021. Two cells remain that I could not match, and I would be grateful if you could tell me the settings used for them.

**1. Univariate ETTh2, horizon 96, DLinear (Table 9: 0.131 / 0.279)**

The script `scripts/EXP-LongForecasting/DLinear/univariate/etth2.sh` (lr 0.005, batch 32) gives 0.1329 / 0.2808 at commit 04ffd8d. A sweep over 6 learning rates × 3 batch sizes found two settings that match the table:

| lr | batch | MSE | MAE |
|---|---|---|---|
| 0.05 | 32 | 0.1308 | 0.2792 |
| 0.005 | 16 | 0.1311 | 0.2794 |

Which one (or another) was used? The other three horizons match with the script as it is.

**2. NLinear, Exchange, horizon 720 (Table 2: 1.033 / 0.780)**

At commit 7f7a25b with `scripts/EXP-LongForecasting/Linear/exchange_rate.sh` and `model_name=NLinear`, I get 0.9430 / 0.7283. The other three Exchange horizons match. I also tried:

- 6 learning rates (0.05 to 0.0001) with batch 32, and 3 learning rates (0.05, 0.005, 0.0005) with batch 8 and 16: MSE 0.883 to 1.123, MAE 0.703 to 0.796
- seeds 2021 to 2025: 0.943 to 0.958 / 0.728 to 0.734
- look-back 96, 192, 512, 720: 0.900 to 1.015 / 0.716 to 0.752

None gives MSE 1.033 and MAE 0.780 together. Could you share the setting or code version for this cell?

**A small note on LTSF-Benchmark.md**

In the second multivariate table (Electricity, Exchange, Traffic, Weather, ILI), the rows labeled "DLinear" and "NLinear" seem to be swapped relative to Table 2 of the paper. For example, ILI-24 2.215 / 1.081 and Exchange-96 0.081 / 0.203 are DLinear in the paper but appear in the NLinear row of the benchmark file.

All logs and scripts are here: https://github.com/an-junhyung/DLinear-colab-reproduction (folder `results/cause_investigation`).

Thank you!

---

## 올리기 전에 정할 것

- 저장소 링크: 지금 저장소는 비공개다. 링크를 넣으려면 공개로 바꾸거나, 마지막 문단을 지우고 올린다.
- 두 질문을 issue 하나로 묶을지 나눌지. 하나로 묶는 편이 답을 받기 쉽다고 봤다.
