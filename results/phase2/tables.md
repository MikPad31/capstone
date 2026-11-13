# Phase 2 results
_generated 2026-11-01 20:56 from output/phase2/_
## Primary specification
All levels and horizons scored on the same 385,119 bond-months (2015-01 to 2021-11, 83 forecast-origin months): those present in every cell. Forecasts projected onto each target's constraint (L2 demeaned within issuer-month, IC averaged within it, L0/L1 unchanged). Significance vs a zero forecast (modified DM): * 10%, ** 5%, *** 1%.


### OOS R² vs zero forecast

|                | 1         | 3      | 12      |
|:---------------|:----------|:-------|:--------|
| ('ols', 'L0')  | +0.74%    | -0.02% | -16.26% |
| ('ols', 'L1')  | -0.16%    | -0.94% | -7.63%  |
| ('ols', 'IC')  | -0.59%    | -1.07% | -1.73%  |
| ('ols', 'L2')  | +1.60%*** | +1.03% | +0.71%  |
| ('enet', 'L0') | +1.31%    | +0.83% | -3.57%  |
| ('enet', 'L1') | +0.34%    | +0.02% | +4.48%  |
| ('enet', 'IC') | +0.29%    | -0.12% | +5.12%  |
| ('enet', 'L2') | +1.42%*** | +0.94% | +0.85%  |
| ('rf', 'L0')   | +0.89%    | +0.55% | -6.72%  |
| ('rf', 'L1')   | -0.13%    | -1.55% | +4.17%  |
| ('rf', 'IC')   | -0.66%    | -0.65% | +5.12%  |
| ('rf', 'L2')   | +1.29%*** | +0.74% | +0.48%  |

### OOS R² vs prevailing training mean

|                |       1 |       3 |      12 |
|:---------------|--------:|--------:|--------:|
| ('ols', 'L0')  | -0.0018 | -0.0160 | -0.1041 |
| ('ols', 'L1')  | -0.0047 | -0.0156 | -0.1169 |
| ('ols', 'IC')  | -0.0096 | -0.0174 | -0.0646 |
| ('ols', 'L2')  | +0.0160 | +0.0103 | +0.0071 |
| ('enet', 'L0') | +0.0040 | -0.0075 | +0.0165 |
| ('enet', 'L1') | +0.0003 | -0.0059 | +0.0088 |
| ('enet', 'IC') | -0.0007 | -0.0079 | +0.0071 |
| ('enet', 'L2') | +0.0142 | +0.0094 | +0.0085 |
| ('rf', 'L0')   | -0.0002 | -0.0102 | -0.0135 |
| ('rf', 'L1')   | -0.0043 | -0.0216 | +0.0056 |
| ('rf', 'IC')   | -0.0103 | -0.0131 | +0.0071 |
| ('rf', 'L2')   | +0.0129 | +0.0074 | +0.0048 |

### Modified Diebold–Mariano tests between forecasts

Positive stat: the first-named model (for 'all vs within', the 341-predictor set) has lower MSE.

| test                | level   |   horizon | stat     |   p_value |   n_months |   lags |
|:--------------------|:--------|----------:|:---------|----------:|-----------:|-------:|
| enet vs ols         | IC      |         1 | +1.29    |     0.198 |         83 |      3 |
| enet vs ols         | IC      |         3 | +0.54    |     0.587 |         83 |      3 |
| enet vs ols         | IC      |        12 | +0.78    |     0.436 |         83 |     11 |
| enet vs ols         | L0      |         1 | +0.81    |     0.418 |         83 |      3 |
| enet vs ols         | L0      |         3 | +0.48    |     0.634 |         83 |      3 |
| enet vs ols         | L0      |        12 | +2.31**  |     0.021 |         83 |     11 |
| enet vs ols         | L1      |         1 | +0.82    |     0.413 |         83 |      3 |
| enet vs ols         | L1      |         3 | +0.57    |     0.567 |         83 |      3 |
| enet vs ols         | L1      |        12 | +1.36    |     0.174 |         83 |     11 |
| enet vs ols         | L2      |         1 | -0.99    |     0.323 |         83 |      3 |
| enet vs ols         | L2      |         3 | -0.12    |     0.901 |         83 |      3 |
| enet vs ols         | L2      |        12 | +0.11    |     0.914 |         83 |     11 |
| enet: all vs within | L2      |         1 | +3.82*** |     0     |         83 |      3 |
| enet: all vs within | L2      |         3 | +1.04    |     0.299 |         83 |      3 |
| enet: all vs within | L2      |        12 | +1.16    |     0.247 |         83 |     11 |
| ols: all vs within  | L2      |         1 | +1.65*   |     0.099 |         83 |      3 |
| ols: all vs within  | L2      |         3 | +0.27    |     0.79  |         83 |      3 |
| ols: all vs within  | L2      |        12 | +0.35    |     0.73  |         83 |     11 |
| rf vs enet          | IC      |         1 | -1.19    |     0.235 |         83 |      3 |
| rf vs enet          | IC      |         3 | -0.48    |     0.63  |         83 |      3 |
| rf vs enet          | IC      |        12 | -0.00    |     0.996 |         83 |     11 |
| rf vs enet          | L0      |         1 | -0.22    |     0.824 |         83 |      3 |
| rf vs enet          | L0      |         3 | -0.23    |     0.82  |         83 |      3 |
| rf vs enet          | L0      |        12 | -1.11    |     0.267 |         83 |     11 |
| rf vs enet          | L1      |         1 | -1.61    |     0.107 |         83 |      3 |
| rf vs enet          | L1      |         3 | -1.89*   |     0.058 |         83 |      3 |
| rf vs enet          | L1      |        12 | -0.14    |     0.885 |         83 |     11 |
| rf vs enet          | L2      |         1 | -0.50    |     0.615 |         83 |      3 |
| rf vs enet          | L2      |         3 | -0.50    |     0.619 |         83 |      3 |
| rf vs enet          | L2      |        12 | -0.59    |     0.556 |         83 |     11 |
| rf vs ols           | IC      |         1 | -0.05    |     0.958 |         83 |      3 |
| rf vs ols           | IC      |         3 | +0.16    |     0.87  |         83 |      3 |
| rf vs ols           | IC      |        12 | +0.92    |     0.357 |         83 |     11 |
| rf vs ols           | L0      |         1 | +0.18    |     0.855 |         83 |      3 |
| rf vs ols           | L0      |         3 | +0.28    |     0.78  |         83 |      3 |
| rf vs ols           | L0      |        12 | +2.35**  |     0.019 |         83 |     11 |
| rf vs ols           | L1      |         1 | +0.07    |     0.948 |         83 |      3 |
| rf vs ols           | L1      |         3 | -0.32    |     0.746 |         83 |      3 |
| rf vs ols           | L1      |        12 | +1.56    |     0.118 |         83 |     11 |
| rf vs ols           | L2      |         1 | -0.98    |     0.325 |         83 |      3 |
| rf vs ols           | L2      |         3 | -0.42    |     0.674 |         83 |      3 |
| rf vs ols           | L2      |        12 | +0.01    |     0.995 |         83 |     11 |
| rf: all vs within   | L2      |         1 | -1.14    |     0.255 |         83 |      3 |
| rf: all vs within   | L2      |         3 | -2.71*** |     0.007 |         83 |      3 |
| rf: all vs within   | L2      |        12 | +0.35    |     0.728 |         83 |     11 |

### Replication inside the pipeline: L0, h=1, OOS R² vs zero (raw forecasts)

| model   |   h1_sample |   h1_sample_to_2021-11 |   primary |
|:--------|------------:|-----------------------:|----------:|
| ols     |     -0.0255 |                -0.0020 |   +0.0074 |
| enet    |     -0.0179 |                +0.0096 |   +0.0131 |
| rf      |     -0.0192 |                +0.0068 |   +0.0089 |

Bond-months: h1_sample 510,510, h1_sample_to_2021-11 447,265, primary 385,119. RF vs OLS (DM, positive = RF better): h1_sample +0.59 (p=0.552), h1_sample_to_2021-11 +0.64 (p=0.522), primary +0.18 (p=0.855).

### L2 OOS R² vs zero by sub-period

|                        |       1 |       3 |      12 |
|:-----------------------|--------:|--------:|--------:|
| ('ols', 'pre_covid')   | +0.0231 | +0.0038 | -0.0634 |
| ('ols', 'covid')       | +0.0065 | +0.0188 | +0.0633 |
| ('ols', 'post_covid')  | +0.0616 | +0.0162 | +0.2293 |
| ('enet', 'pre_covid')  | +0.0216 | +0.0085 | -0.0061 |
| ('enet', 'covid')      | +0.0044 | +0.0110 | +0.0017 |
| ('enet', 'post_covid') | +0.0590 | +0.0073 | +0.1563 |
| ('rf', 'pre_covid')    | +0.0169 | +0.0017 | -0.0086 |
| ('rf', 'covid')        | +0.0077 | +0.0129 | +0.0107 |
| ('rf', 'post_covid')   | +0.0380 | +0.0314 | +0.0727 |

Forecast-origin months per sub-period:

| period     |   1 |   3 |   12 |
|:-----------|----:|----:|-----:|
| pre_covid  |  60 |  60 |   60 |
| covid      |  12 |  12 |   12 |
| post_covid |  11 |  11 |   11 |

## Appendix: each horizon's own common sample, raw forecasts

### OOS R² vs zero forecast

|                |       1 |       3 |      12 |
|:---------------|--------:|--------:|--------:|
| ('ols', 'L0')  | -0.0255 | -0.0987 | -0.1626 |
| ('ols', 'L1')  | -0.0111 | -0.0382 | -0.0763 |
| ('ols', 'IC')  | -0.0169 | -0.0434 | -0.0779 |
| ('ols', 'L2')  | +0.0051 | -0.0076 | -0.0305 |
| ('enet', 'L0') | -0.0179 | -0.0755 | -0.0357 |
| ('enet', 'L1') | -0.0001 | -0.0098 | +0.0448 |
| ('enet', 'IC') | -0.0013 | -0.0125 | +0.0501 |
| ('enet', 'L2') | +0.0082 | +0.0047 | +0.0069 |
| ('rf', 'L0')   | -0.0192 | -0.0663 | -0.0672 |
| ('rf', 'L1')   | -0.0046 | -0.0258 | +0.0417 |
| ('rf', 'IC')   | -0.0095 | -0.0165 | +0.0489 |
| ('rf', 'L2')   | +0.0089 | +0.0035 | +0.0019 |

### OOS R² vs prevailing training mean

|                |       1 |       3 |      12 |
|:---------------|--------:|--------:|--------:|
| ('ols', 'L0')  | -0.0108 | -0.0452 | -0.1041 |
| ('ols', 'L1')  | -0.0121 | -0.0380 | -0.1169 |
| ('ols', 'IC')  | -0.0182 | -0.0431 | -0.1280 |
| ('ols', 'L2')  | +0.0051 | -0.0076 | -0.0305 |
| ('enet', 'L0') | -0.0032 | -0.0232 | +0.0165 |
| ('enet', 'L1') | -0.0010 | -0.0096 | +0.0088 |
| ('enet', 'IC') | -0.0025 | -0.0123 | +0.0060 |
| ('enet', 'L2') | +0.0082 | +0.0047 | +0.0069 |
| ('rf', 'L0')   | -0.0046 | -0.0143 | -0.0135 |
| ('rf', 'L1')   | -0.0055 | -0.0256 | +0.0056 |
| ('rf', 'IC')   | -0.0107 | -0.0163 | +0.0047 |
| ('rf', 'L2')   | +0.0089 | +0.0035 | +0.0019 |

### OOS R² vs zero, forecasts projected onto the level's constraint

|                |       1 |       3 |      12 |
|:---------------|--------:|--------:|--------:|
| ('ols', 'L0')  | -0.0255 | -0.0987 | -0.1626 |
| ('ols', 'L1')  | -0.0111 | -0.0382 | -0.0763 |
| ('ols', 'IC')  | -0.0139 | -0.0346 | -0.0173 |
| ('ols', 'L2')  | +0.0124 | +0.0054 | +0.0071 |
| ('enet', 'L0') | -0.0179 | -0.0755 | -0.0357 |
| ('enet', 'L1') | -0.0001 | -0.0098 | +0.0448 |
| ('enet', 'IC') | -0.0008 | -0.0113 | +0.0512 |
| ('enet', 'L2') | +0.0121 | +0.0079 | +0.0085 |
| ('rf', 'L0')   | -0.0192 | -0.0663 | -0.0672 |
| ('rf', 'L1')   | -0.0046 | -0.0258 | +0.0417 |
| ('rf', 'IC')   | -0.0087 | -0.0154 | +0.0512 |
| ('rf', 'L2')   | +0.0106 | +0.0054 | +0.0048 |

### Scored bond-months (identical across levels on the common sample)

| level   |      1 |      3 |     12 |
|:--------|-------:|-------:|-------:|
| L0      | 510510 | 486610 | 385119 |
| L1      | 510510 | 486610 | 385119 |
| IC      | 510510 | 486610 | 385119 |
| L2      | 510510 | 486610 | 385119 |

### Modified Diebold–Mariano tests

Positive stat: the first-named model (or, for 'vs zero', the model) has lower MSE. Newey–West, lags = max(h−1, ⌊4(T/100)^(2/9)⌋). * 10%, ** 5%, *** 1%.

| test                | level   |   horizon | predictor_set   | stat      |   p_value |   n_months |   lags |
|:--------------------|:--------|----------:|:----------------|:----------|----------:|-----------:|-------:|
| enet vs ols         | IC      |         1 | all             | +2.09**   |     0.037 |         94 |      3 |
| enet vs ols         | IC      |         3 | all             | +1.29     |     0.197 |         92 |      3 |
| enet vs ols         | IC      |        12 | all             | +1.44     |     0.15  |         83 |     11 |
| enet vs ols         | L0      |         1 | all             | +1.23     |     0.217 |         94 |      3 |
| enet vs ols         | L0      |         3 | all             | +1.16     |     0.246 |         92 |      3 |
| enet vs ols         | L0      |        12 | all             | +2.31**   |     0.021 |         83 |     11 |
| enet vs ols         | L1      |         1 | all             | +1.53     |     0.125 |         94 |      3 |
| enet vs ols         | L1      |         3 | all             | +1.14     |     0.253 |         92 |      3 |
| enet vs ols         | L1      |        12 | all             | +1.36     |     0.174 |         83 |     11 |
| enet vs ols         | L2      |         1 | all             | +1.62     |     0.105 |         94 |      3 |
| enet vs ols         | L2      |         1 | within          | +1.21     |     0.224 |         94 |      3 |
| enet vs ols         | L2      |         3 | all             | +1.92*    |     0.054 |         92 |      3 |
| enet vs ols         | L2      |         3 | within          | +1.62     |     0.104 |         92 |      3 |
| enet vs ols         | L2      |        12 | all             | +1.05     |     0.295 |         83 |     11 |
| enet vs ols         | L2      |        12 | within          | +0.96     |     0.336 |         83 |     11 |
| enet vs zero        | IC      |         1 | all             | -0.14     |     0.891 |         94 |      3 |
| enet vs zero        | IC      |         3 | all             | -0.39     |     0.694 |         92 |      3 |
| enet vs zero        | IC      |        12 | all             | +0.37     |     0.709 |         83 |     11 |
| enet vs zero        | L0      |         1 | all             | -0.70     |     0.485 |         94 |      3 |
| enet vs zero        | L0      |         3 | all             | -1.09     |     0.278 |         92 |      3 |
| enet vs zero        | L0      |        12 | all             | -0.06     |     0.953 |         83 |     11 |
| enet vs zero        | L1      |         1 | all             | -0.04     |     0.967 |         94 |      3 |
| enet vs zero        | L1      |         3 | all             | -0.33     |     0.739 |         92 |      3 |
| enet vs zero        | L1      |        12 | all             | +0.39     |     0.696 |         83 |     11 |
| enet vs zero        | L2      |         1 | all             | +1.86*    |     0.063 |         94 |      3 |
| enet vs zero        | L2      |         1 | within          | +1.74*    |     0.082 |         94 |      3 |
| enet vs zero        | L2      |         3 | all             | +0.43     |     0.668 |         92 |      3 |
| enet vs zero        | L2      |         3 | within          | +0.40     |     0.691 |         92 |      3 |
| enet vs zero        | L2      |        12 | all             | +0.62     |     0.538 |         83 |     11 |
| enet vs zero        | L2      |        12 | within          | +0.61     |     0.542 |         83 |     11 |
| enet: all vs within | L2      |         1 | all|within      | +2.47**   |     0.013 |         94 |      3 |
| enet: all vs within | L2      |         3 | all|within      | +0.58     |     0.565 |         92 |      3 |
| enet: all vs within | L2      |        12 | all|within      | +1.26     |     0.207 |         83 |     11 |
| ols vs zero         | IC      |         1 | all             | -1.14     |     0.256 |         94 |      3 |
| ols vs zero         | IC      |         3 | all             | -0.87     |     0.386 |         92 |      3 |
| ols vs zero         | IC      |        12 | all             | -0.40     |     0.692 |         83 |     11 |
| ols vs zero         | L0      |         1 | all             | -0.96     |     0.337 |         94 |      3 |
| ols vs zero         | L0      |         3 | all             | -1.26     |     0.209 |         92 |      3 |
| ols vs zero         | L0      |        12 | all             | -0.58     |     0.565 |         83 |     11 |
| ols vs zero         | L1      |         1 | all             | -0.78     |     0.438 |         94 |      3 |
| ols vs zero         | L1      |         3 | all             | -0.80     |     0.426 |         92 |      3 |
| ols vs zero         | L1      |        12 | all             | -0.44     |     0.663 |         83 |     11 |
| ols vs zero         | L2      |         1 | all             | +0.80     |     0.426 |         94 |      3 |
| ols vs zero         | L2      |         1 | constant        | -0.84     |     0.403 |         94 |      3 |
| ols vs zero         | L2      |         1 | within          | +0.82     |     0.413 |         94 |      3 |
| ols vs zero         | L2      |         3 | all             | -0.71     |     0.475 |         92 |      3 |
| ols vs zero         | L2      |         3 | constant        | +0.23     |     0.814 |         92 |      3 |
| ols vs zero         | L2      |         3 | within          | -0.61     |     0.543 |         92 |      3 |
| ols vs zero         | L2      |        12 | all             | -0.85     |     0.395 |         83 |     11 |
| ols vs zero         | L2      |        12 | constant        | +1.12     |     0.262 |         83 |     11 |
| ols vs zero         | L2      |        12 | within          | -0.78     |     0.438 |         83 |     11 |
| ols: all vs within  | L2      |         1 | all|within      | -0.22     |     0.827 |         94 |      3 |
| ols: all vs within  | L2      |         3 | all|within      | -1.10     |     0.272 |         92 |      3 |
| ols: all vs within  | L2      |        12 | all|within      | -1.01     |     0.315 |         83 |     11 |
| rf vs enet          | IC      |         1 | all             | -1.15     |     0.248 |         94 |      3 |
| rf vs enet          | IC      |         3 | all             | -0.37     |     0.708 |         92 |      3 |
| rf vs enet          | IC      |        12 | all             | -0.07     |     0.942 |         83 |     11 |
| rf vs enet          | L0      |         1 | all             | -0.08     |     0.939 |         94 |      3 |
| rf vs enet          | L0      |         3 | all             | +0.80     |     0.425 |         92 |      3 |
| rf vs enet          | L0      |        12 | all             | -1.11     |     0.267 |         83 |     11 |
| rf vs enet          | L1      |         1 | all             | -1.55     |     0.12  |         94 |      3 |
| rf vs enet          | L1      |         3 | all             | -1.92*    |     0.055 |         92 |      3 |
| rf vs enet          | L1      |        12 | all             | -0.14     |     0.885 |         83 |     11 |
| rf vs enet          | L2      |         1 | all             | +0.29     |     0.772 |         94 |      3 |
| rf vs enet          | L2      |         1 | within          | +1.05     |     0.292 |         94 |      3 |
| rf vs enet          | L2      |         3 | all             | -0.22     |     0.823 |         92 |      3 |
| rf vs enet          | L2      |         3 | within          | +0.79     |     0.432 |         92 |      3 |
| rf vs enet          | L2      |        12 | all             | -0.85     |     0.396 |         83 |     11 |
| rf vs enet          | L2      |        12 | within          | -1.23     |     0.22  |         83 |     11 |
| rf vs ols           | IC      |         1 | all             | +0.62     |     0.537 |         94 |      3 |
| rf vs ols           | IC      |         3 | all             | +0.91     |     0.362 |         92 |      3 |
| rf vs ols           | IC      |        12 | all             | +1.68*    |     0.092 |         83 |     11 |
| rf vs ols           | L0      |         1 | all             | +0.59     |     0.552 |         94 |      3 |
| rf vs ols           | L0      |         3 | all             | +1.50     |     0.133 |         92 |      3 |
| rf vs ols           | L0      |        12 | all             | +2.35**   |     0.019 |         83 |     11 |
| rf vs ols           | L1      |         1 | all             | +0.93     |     0.351 |         94 |      3 |
| rf vs ols           | L1      |         3 | all             | +0.49     |     0.625 |         92 |      3 |
| rf vs ols           | L1      |        12 | all             | +1.56     |     0.118 |         83 |     11 |
| rf vs ols           | L2      |         1 | all             | +1.01     |     0.313 |         94 |      3 |
| rf vs ols           | L2      |         1 | constant        | -13.43*** |     0     |         94 |      3 |
| rf vs ols           | L2      |         1 | within          | +1.33     |     0.183 |         94 |      3 |
| rf vs ols           | L2      |         3 | all             | +1.55     |     0.121 |         92 |      3 |
| rf vs ols           | L2      |         3 | constant        | -14.58*** |     0     |         92 |      3 |
| rf vs ols           | L2      |         3 | within          | +2.02**   |     0.043 |         92 |      3 |
| rf vs ols           | L2      |        12 | all             | +0.93     |     0.351 |         83 |     11 |
| rf vs ols           | L2      |        12 | constant        | -5.96***  |     0     |         83 |     11 |
| rf vs ols           | L2      |        12 | within          | +0.83     |     0.406 |         83 |     11 |
| rf vs zero          | IC      |         1 | all             | -0.63     |     0.53  |         94 |      3 |
| rf vs zero          | IC      |         3 | all             | -0.48     |     0.632 |         92 |      3 |
| rf vs zero          | IC      |        12 | all             | +0.33     |     0.743 |         83 |     11 |
| rf vs zero          | L0      |         1 | all             | -0.74     |     0.459 |         94 |      3 |
| rf vs zero          | L0      |         3 | all             | -0.99     |     0.321 |         92 |      3 |
| rf vs zero          | L0      |        12 | all             | -0.17     |     0.862 |         83 |     11 |
| rf vs zero          | L1      |         1 | all             | -0.43     |     0.669 |         94 |      3 |
| rf vs zero          | L1      |         3 | all             | -0.74     |     0.461 |         92 |      3 |
| rf vs zero          | L1      |        12 | all             | +0.32     |     0.752 |         83 |     11 |
| rf vs zero          | L2      |         1 | all             | +3.29***  |     0.001 |         94 |      3 |
| rf vs zero          | L2      |         1 | constant        | -13.43*** |     0     |         94 |      3 |
| rf vs zero          | L2      |         1 | within          | +3.56***  |     0     |         94 |      3 |
| rf vs zero          | L2      |         3 | all             | +0.47     |     0.636 |         92 |      3 |
| rf vs zero          | L2      |         3 | constant        | -14.58*** |     0     |         92 |      3 |
| rf vs zero          | L2      |         3 | within          | +0.86     |     0.391 |         92 |      3 |
| rf vs zero          | L2      |        12 | all             | +0.11     |     0.916 |         83 |     11 |
| rf vs zero          | L2      |        12 | constant        | -5.96***  |     0     |         83 |     11 |
| rf vs zero          | L2      |        12 | within          | -0.35     |     0.728 |         83 |     11 |
| rf: all vs within   | L2      |         1 | all|within      | -0.78     |     0.434 |         94 |      3 |
| rf: all vs within   | L2      |         3 | all|within      | -2.05**   |     0.04  |         92 |      3 |
| rf: all vs within   | L2      |        12 | all|within      | +0.72     |     0.474 |         83 |     11 |

### DNR shipped forecasts on our h=1 test rows

| level   | dnr_model          |      n |   coverage |   r2_zero_our_label |   r2_zero_raw_return |
|:--------|:-------------------|-------:|-----------:|--------------------:|---------------------:|
| L0      | elastic_net        | 510510 |    +1.0000 |             -0.0085 |              -0.0051 |
| L0      | extra_trees        | 510510 |    +1.0000 |             -0.0028 |              -0.0005 |
| L0      | full_ensemble      | 510510 |    +1.0000 |             -0.0029 |              -0.0008 |
| L0      | lasso              | 510510 |    +1.0000 |             -0.0073 |              -0.0045 |
| L0      | linear_ensemble    | 510510 |    +1.0000 |             -0.0066 |              -0.0034 |
| L0      | neural_net         | 510510 |    +1.0000 |             -0.0028 |              -0.0014 |
| L0      | nonlinear_ensemble | 510510 |    +1.0000 |             -0.0038 |              -0.0015 |
| L0      | random_forest      | 510510 |    +1.0000 |             -0.0086 |              -0.0043 |
| L0      | ridge              | 510510 |    +1.0000 |             -0.0160 |              -0.0091 |
| L1      | elastic_net        | 510510 |    +1.0000 |             +0.0038 |              +0.0034 |
| L1      | extra_trees        | 510510 |    +1.0000 |             +0.0109 |              +0.0085 |
| L1      | full_ensemble      | 510510 |    +1.0000 |             +0.0093 |              +0.0074 |
| L1      | lasso              | 510510 |    +1.0000 |             +0.0053 |              +0.0042 |
| L1      | linear_ensemble    | 510510 |    +1.0000 |             +0.0048 |              +0.0044 |
| L1      | neural_net         | 510510 |    +1.0000 |             +0.0060 |              +0.0044 |
| L1      | nonlinear_ensemble | 510510 |    +1.0000 |             +0.0082 |              +0.0068 |
| L1      | random_forest      | 510510 |    +1.0000 |             +0.0043 |              +0.0051 |
| L1      | ridge              | 510510 |    +1.0000 |             -0.0060 |              -0.0018 |

### Sub-period robustness

Same out-of-sample predictions, re-scored within blocks of forecast-origin months (pre_covid = 2015-01 to 2019-12, covid = 2020-01 to 2020-12, post_covid = 2021-01 to 2022-11). At h=12 a late-2019 forecast's label already includes 2020 returns.


#### OOS R² vs zero forecast, by sub-period

|                              |       1 |       3 |      12 |
|:-----------------------------|--------:|--------:|--------:|
| ('ols', 'L0', 'pre_covid')   | -0.0291 | -0.0639 | +0.0469 |
| ('ols', 'L0', 'covid')       | +0.0249 | +0.0982 | +0.4311 |
| ('ols', 'L0', 'post_covid')  | -0.0977 | -0.3456 | -0.9030 |
| ('ols', 'L1', 'pre_covid')   | -0.0428 | -0.0941 | -0.3213 |
| ('ols', 'L1', 'covid')       | +0.0120 | +0.0804 | +0.3409 |
| ('ols', 'L1', 'post_covid')  | -0.0478 | -0.3352 | -1.0049 |
| ('ols', 'IC', 'pre_covid')   | -0.0544 | -0.0987 | -0.3175 |
| ('ols', 'IC', 'covid')       | +0.0115 | +0.0835 | +0.3619 |
| ('ols', 'IC', 'post_covid')  | -0.1076 | -0.5322 | -1.6255 |
| ('ols', 'L2', 'pre_covid')   | +0.0063 | -0.0185 | -0.1208 |
| ('ols', 'L2', 'covid')       | -0.0008 | +0.0089 | +0.0534 |
| ('ols', 'L2', 'post_covid')  | +0.0244 | -0.0074 | +0.1888 |
| ('enet', 'L0', 'pre_covid')  | +0.0024 | -0.0195 | +0.2274 |
| ('enet', 'L0', 'covid')      | +0.0187 | +0.0786 | +0.3986 |
| ('enet', 'L0', 'post_covid') | -0.0989 | -0.3204 | -0.8082 |
| ('enet', 'L1', 'pre_covid')  | -0.0028 | -0.0362 | -0.0547 |
| ('enet', 'L1', 'covid')      | +0.0039 | +0.0512 | +0.2513 |
| ('enet', 'L1', 'post_covid') | -0.0193 | -0.1952 | -0.7197 |
| ('enet', 'IC', 'pre_covid')  | -0.0021 | -0.0383 | -0.0618 |
| ('enet', 'IC', 'covid')      | +0.0031 | +0.0501 | +0.2869 |
| ('enet', 'IC', 'post_covid') | -0.0429 | -0.2823 | -1.0759 |
| ('enet', 'L2', 'pre_covid')  | +0.0134 | +0.0030 | -0.0075 |
| ('enet', 'L2', 'covid')      | -0.0002 | +0.0046 | +0.0017 |
| ('enet', 'L2', 'post_covid') | +0.0234 | +0.0165 | +0.1442 |
| ('rf', 'L0', 'pre_covid')    | +0.0313 | -0.0186 | +0.2191 |
| ('rf', 'L0', 'covid')        | +0.0057 | +0.0751 | +0.3648 |
| ('rf', 'L0', 'post_covid')   | -0.1204 | -0.2842 | -0.8875 |
| ('rf', 'L1', 'pre_covid')    | -0.0133 | -0.0684 | -0.0644 |
| ('rf', 'L1', 'covid')        | +0.0024 | +0.0588 | +0.2640 |
| ('rf', 'L1', 'post_covid')   | -0.0198 | -0.2000 | -0.7945 |
| ('rf', 'IC', 'pre_covid')    | -0.0110 | -0.0552 | -0.0709 |
| ('rf', 'IC', 'covid')        | -0.0077 | +0.0648 | +0.3017 |
| ('rf', 'IC', 'post_covid')   | -0.0191 | -0.2692 | -1.1491 |
| ('rf', 'L2', 'pre_covid')    | +0.0116 | -0.0014 | -0.0132 |
| ('rf', 'L2', 'covid')        | +0.0052 | +0.0100 | +0.0104 |
| ('rf', 'L2', 'post_covid')   | +0.0138 | +0.0083 | +0.0690 |

#### OOS R² vs prevailing training mean, by sub-period

|                              |       1 |       3 |      12 |
|:-----------------------------|--------:|--------:|--------:|
| ('ols', 'L0', 'pre_covid')   | -0.0364 | -0.0721 | -0.2162 |
| ('ols', 'L0', 'covid')       | +0.0133 | +0.0425 | +0.3065 |
| ('ols', 'L0', 'post_covid')  | -0.0149 | -0.0685 | -0.0956 |
| ('ols', 'L1', 'pre_covid')   | -0.0449 | -0.0799 | -0.3007 |
| ('ols', 'L1', 'covid')       | +0.0100 | +0.0439 | +0.1976 |
| ('ols', 'L1', 'post_covid')  | -0.0343 | -0.1437 | -0.1659 |
| ('ols', 'IC', 'pre_covid')   | -0.0572 | -0.0813 | -0.2926 |
| ('ols', 'IC', 'covid')       | +0.0090 | +0.0389 | +0.1857 |
| ('ols', 'IC', 'post_covid')  | -0.0855 | -0.2145 | -0.2648 |
| ('ols', 'L2', 'pre_covid')   | +0.0063 | -0.0185 | -0.1208 |
| ('ols', 'L2', 'covid')       | -0.0008 | +0.0089 | +0.0534 |
| ('ols', 'L2', 'post_covid')  | +0.0244 | -0.0074 | +0.1888 |
| ('enet', 'L0', 'pre_covid')  | -0.0047 | -0.0274 | +0.0141 |
| ('enet', 'L0', 'covid')      | +0.0071 | +0.0217 | +0.2669 |
| ('enet', 'L0', 'post_covid') | -0.0160 | -0.0485 | -0.0410 |
| ('enet', 'L1', 'pre_covid')  | -0.0048 | -0.0227 | -0.0383 |
| ('enet', 'L1', 'covid')      | +0.0018 | +0.0135 | +0.0886 |
| ('enet', 'L1', 'post_covid') | -0.0062 | -0.0238 | +0.0000 |
| ('enet', 'IC', 'pre_covid')  | -0.0048 | -0.0219 | -0.0418 |
| ('enet', 'IC', 'covid')      | +0.0006 | +0.0039 | +0.0899 |
| ('enet', 'IC', 'post_covid') | -0.0221 | -0.0164 | +0.0000 |
| ('enet', 'L2', 'pre_covid')  | +0.0134 | +0.0030 | -0.0075 |
| ('enet', 'L2', 'covid')      | -0.0002 | +0.0046 | +0.0017 |
| ('enet', 'L2', 'post_covid') | +0.0234 | +0.0165 | +0.1442 |
| ('rf', 'L0', 'pre_covid')    | +0.0244 | -0.0265 | +0.0035 |
| ('rf', 'L0', 'covid')        | -0.0061 | +0.0179 | +0.2257 |
| ('rf', 'L0', 'post_covid')   | -0.0359 | -0.0197 | -0.0867 |
| ('rf', 'L1', 'pre_covid')    | -0.0153 | -0.0545 | -0.0479 |
| ('rf', 'L1', 'covid')        | +0.0004 | +0.0215 | +0.1040 |
| ('rf', 'L1', 'post_covid')   | -0.0067 | -0.0279 | -0.0435 |
| ('rf', 'IC', 'pre_covid')    | -0.0137 | -0.0385 | -0.0507 |
| ('rf', 'IC', 'covid')        | -0.0102 | +0.0192 | +0.1088 |
| ('rf', 'IC', 'post_covid')   | +0.0012 | -0.0060 | -0.0353 |
| ('rf', 'L2', 'pre_covid')    | +0.0116 | -0.0014 | -0.0132 |
| ('rf', 'L2', 'covid')        | +0.0052 | +0.0100 | +0.0104 |
| ('rf', 'L2', 'post_covid')   | +0.0138 | +0.0083 | +0.0690 |

#### Forecast-origin months per sub-period

| period     |   1 |   3 |   12 |
|:-----------|----:|----:|-----:|
| pre_covid  |  60 |  60 |   60 |
| covid      |  12 |  12 |   12 |
| post_covid |  22 |  20 |   11 |

#### L2 OOS R² vs zero: all 341 predictors vs the 40 within-issuer ones, by sub-period

|                                  |       1 |       3 |      12 |
|:---------------------------------|--------:|--------:|--------:|
| ('ols', 'pre_covid', 'all')      | +0.0063 | -0.0185 | -0.1208 |
| ('ols', 'pre_covid', 'within')   | +0.0068 | -0.0163 | -0.1105 |
| ('ols', 'covid', 'all')          | -0.0008 | +0.0089 | +0.0534 |
| ('ols', 'covid', 'within')       | -0.0008 | +0.0093 | +0.0499 |
| ('ols', 'post_covid', 'all')     | +0.0244 | -0.0074 | +0.1888 |
| ('ols', 'post_covid', 'within')  | +0.0231 | -0.0057 | +0.1810 |
| ('enet', 'pre_covid', 'all')     | +0.0134 | +0.0030 | -0.0075 |
| ('enet', 'pre_covid', 'within')  | +0.0120 | +0.0021 | -0.0075 |
| ('enet', 'covid', 'all')         | -0.0002 | +0.0046 | +0.0017 |
| ('enet', 'covid', 'within')      | -0.0001 | +0.0046 | +0.0017 |
| ('enet', 'post_covid', 'all')    | +0.0234 | +0.0165 | +0.1442 |
| ('enet', 'post_covid', 'within') | +0.0227 | +0.0165 | +0.1428 |
| ('rf', 'pre_covid', 'all')       | +0.0116 | -0.0014 | -0.0132 |
| ('rf', 'pre_covid', 'within')    | +0.0114 | +0.0002 | -0.0226 |
| ('rf', 'covid', 'all')           | +0.0052 | +0.0100 | +0.0104 |
| ('rf', 'covid', 'within')        | +0.0046 | +0.0150 | +0.0096 |
| ('rf', 'post_covid', 'all')      | +0.0138 | +0.0083 | +0.0690 |
| ('rf', 'post_covid', 'within')   | +0.0243 | +0.0180 | +0.0889 |
