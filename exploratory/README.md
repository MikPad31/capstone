# exploratory/

One line per script: what it asked, what it found.

- `key_choice_exploration.py`:
    is `gvkey` or `issuer_cusip` the right issuer key for L2?
    Settled on `gvkey` (consolidated parent firm), the key every Phase 2 target uses (IC, L2,
    the common sample). Functions in `src/decompose.py` still take either key.

- `key_choice_r2.py`:
    companion to the above: what the key choice does to exposure, the decomposition, the folds
    and power. Findings were not written up here; rerun it for the numbers.

- `splits_prototype.py`:
    does an `h`-month embargo remove the overlap between a fold's
    training labels and its test labels? Yes, exactly. `s <= T-1` leaks `h-1` calendar months
    per fold; `s <= T-h` leaks none, costing `h` unusable signal months at the panel tail and
    `h-1` training months at the head. Promoted to `src/splits.py`.

    Label convention (`t` is labelled by returns over `t+1 … t+h`) checked against
    `dnr_ml_predictions/README.txt`, where `realized_return_date` is `t+1` for `signal_date`
    `t`. The leakage checks move with it and so cannot catch it; pinned in `test/test_splits.py`.

    Counts below assume `oos_start=2015-01`, since fixed in `config.OOS_START`.

  ```
  [verify] PASS — naive splitter at h=12 leaks exactly h-1=11 calendar months in every one of its 7 folds (got [11])
  [verify] PASS — embargoed splitter at h=12 leaks 0 calendar months in all 7 folds
  [report] leaked calendar months per fold, naive → embargoed — h=1: 0 → 0, h=3: 2 → 0, h=12: 11 → 0
  [verify] PASS — V3 invariants hold across 36 parameter combinations (2043 folds); 0 violation(s)
  [verify] PASS — h=12: 7 folds == ceil(83 usable OOS months / 12)
  ```

- `targets_prototype.py`:
    does compounding `t+1 … t+h` within `cusip` reproduce the shipped realized return at
    `h=1`, and what does it cost in sample size at `h=3`/`h=12`? Yes, exactly once; the
    compounding window is corrected. `retx_realized_return`/`retxrf_realized_return` are
    already one-month-forward (`t` covers `t` to `t+1`), so the window is
    `[t, ..., t+h-1]`, not `[t+1, ..., t+h]`.

    Tail loss is `h-1` months, not `h`: h=1 loses nothing to the grid edge; h=3/h=12
    lose 2/11 months.

    Finding: the panel vs predictions disagreement previously cited as "664 rows (0.06%),
    5 cusips" (never independently verified) does not reproduce for
    `retx_realized_return`/`retxrf_realized_return` after `load_panel`'s join -- 0
    missing rows for both targets across the full 1,102,569-row panel.

```
[verify] PASS — grid round-trip: 1,102,569 non-null cells vs 1,102,569 input rows
[verify] PASS — h=1 forward_return(retx_realized_return) reproduces its input exactly on 1102569 rows (max abs diff 0.0)
[verify] PASS — h=1 one-sided rows: 0 fwd-only, 0 input-only
[verify] PASS — h=1 forward_return(retxrf_realized_return) reproduces its input exactly on 1102569 rows (max abs diff 0.0)
[verify] PASS — h=1 one-sided rows: 0 fwd-only, 0 input-only
[report] retx_realized_return h=1: 100.0% labelled overall; 0 missing after join (0.0000%)
[report] retx_realized_return h=3: 96.2% labelled overall
[report] retx_realized_return h=12: 79.8% labelled overall
[report] retxrf_realized_return h=1: 100.0% labelled overall; 0 missing after join (0.0000%)
[report] retxrf_realized_return h=3: 96.2% labelled overall
[report] retxrf_realized_return h=12: 79.8% labelled overall
```

- `target_pipeline.py`:
    does the decomposition, fitted fold by fold with the embargo, produce an OOS R², and does the
    embargo cost anything measurable? Yes to both. Nine cells (3 levels × h ∈ {1, 3, 12}, OLS, one
    run): predictability rises at every step down the decomposition and falls with horizon, no
    reversals. Only L2 at h=1 beats a zero forecast.

    L0 and L1 are scored on identical rows at every horizon, so L0→L1 is a pure change of target;
    only the step to L2 and the horizon move the sample. Rescoring against the demeaned benchmark
    from the saved sums (no refit) leaves the ordering unchanged and widens the L1→L2 gap to
    +0.074 at h=12. L2 is identical under both conventions, its target's mean being zero by
    construction.

    Missingness is 0.0000% only because the panel ships pre-imputed to 0.0 at the neutral rank;
    sparsity shows up solely as exact zeros. No imputation step exists, so none can leak.

    Checks, 0 failures: V2 54, V3 474 across 69 folds, V4 54, one finite-predictor guard per fold,
    and the `r2_from_sums` cross-check in all nine cells. Representative lines below; the full log
    is ~700 lines.

    Counts assume `oos_start=2015-01`, since fixed in `config.OOS_START`.

```
OOS R^2 (zero benchmark), levels x horizons:
horizon      1       3       12
level
L0      -0.0237 -0.0936 -0.1390
L1      -0.0094 -0.0354 -0.0544
L2       0.0051 -0.0076 -0.0305
n (scored bond-months):
horizon      1       3       12
level
L0       527436  502977  399175
L1       527436  502977  399175
L2       510510  486610  385119
[report] predictor missingness: 0.0000% (0 of 375,976,029 cells NaN); 0 columns affected.
[report] Predictor exact zeros: 8.3485% (31,388,258 of 375,976,029 cells are exactly zero)
[report] L2 h=1: |E| = 1,037,576 bond-months (94.1% of 1,102,569)
[report] L2 h=3: |E| = 997,100 bond-months (90.4% of 1,102,569)
[report] L2 h=12: |E| = 823,083 bond-months (74.7% of 1,102,569)
[verify] PASS — L0 h=1: eligibility exactly equals the non-missing label mask
[verify] PASS — retx_realized_return h=1: stored label equals the raw label winsorized by formation month at 1/99
[verify] PASS — L0 h=1: no duplicated (date, cusip) pairs (0 found)
[verify] PASS — r2_from_sums (-0.0236566290) matches oos_r2 (-0.0236566290) for benchmark 0.0
[report] Negative control results for level L0, horizon 12 months: Embargoed R^2 = -0.1390, Embargoed train rows = 4,154,187
[report] Negative control results for level L0, horizon 12 months: Naive R^2 = -0.0928, Naive train rows = 4,516,838
done in 956.8s. peak rss=5.56GB
```

- `data_summary.py`:
    what does the data look like, for the report's data section and the slides? 1,102,569
    bond-months, 19,768 bonds, 2,030 issuers, 245 months (2002-07 to 2022-11); 94.1% of
    bond-months share an issuer-month with another bond (median 3 bonds per issuer-month).
    Predictors have no missing values and lie in [-1, 1]. On the common sample, monthly SD is
    3.21% (L0), 2.99% (L1), 2.65% (IC) and 1.38% (L2); L2's share of L1's sum of squares falls
    with horizon, 21.3% / 16.8% / 13.0% at h=1/3/12. IC and L2 rebuilt here match the saved
    test-period labels exactly at every horizon. Writes `output/data_summary/`; ~20 s.

- `within_issuer_predictors.py`:
    which predictors vary across an issuer's bonds in the same month? Exactly 301 of 341 never
    vary within a (`gvkey`, month); 40 vary in >= 5% of multi-bond issuer-months (38 of them in
    > 50%). No predictor sits in between, so the split is not threshold-sensitive. Writes
    `output/within_issuer_predictors.csv`, read by `phase2_grid.py`.

- `phase2_grid.py` / `phase2_results.py` (harness promoted to `src/harness.py`, models in
    `src/models.py`, DM test in `src/inference.py`; full run: `bash scripts/run_phase2.sh`):
    OLS, Elastic Net, Random Forest (LightGBM `boosting="rf"`) x L0/L1/IC/L2 x h=1/3/12 on the
    common sample (same L2-eligible rows for every level), plus L2 with the 40 within-issuer
    predictors. EN and RF tune on the last 36 training months behind an h-month embargo.

    Only L2 at h=1 beats a zero forecast significantly: RF +0.0089 (DM p=0.001), EN +0.0082
    (p=0.06, so 10% level only). L1/IC at h=12 reach +0.04-0.05 vs zero but < 0.01 vs the
    prevailing training mean, and are insignificant. EN and RF have higher R^2 than OLS in every
    cell of the all-predictor grid, but the DM difference is significant at 5% only for EN
    (L0 h=12, IC h=1) and RF (L0 h=12); RF never differs significantly from EN at 5%.

    RF's chosen depth depends on the target. For L0/L1/IC it picks depth 1 in 43 of 69 folds,
    mostly at h=3 and h=12 (at h=1 the choice varies across 1-6). For L2 at h=1 it picks depth 6,
    the grid maximum, in 6 of 8 folds (3 of 8 at h=3). Since depth 6 binds for L2, the grid should
    extend deeper before reading anything into L2 nonlinearity.

    For L2, the 40 within-issuer predictors match all 341 under RF at h=1 and h=12 and beat
    them at h=3 (DM p=0.04); under EN, all 341 do better at h=1 only (p=0.013). Demeaning L2
    forecasts within issuer-month raises R^2 for every model at every horizon. DNR's shipped
    forecasts are also negative at L0 on our h=1 rows; at L1 they beat ours (best +0.0109 vs
    our EN -0.0001), cause not yet established.

    Checks: the original `target_pipeline.py` reproduces the midterm OLS table exactly (all 9
    cells); the new harness reproduces it on the two cells checked (L0 and L2 at h=1, midterm
    samples). L1 = IC + L2 row by row for all models and horizons (max gap 0). With only
    issuer-constant predictors, L2 R^2 is exactly 0 for OLS and within 4e-7 of 0 for RF.

```
OOS R^2 (zero benchmark), common sample, all predictors:
           h=1      h=3      h=12
ols  L2  +0.0051  -0.0076  -0.0305
enet L2  +0.0082  +0.0047  +0.0069
rf   L2  +0.0089  +0.0035  +0.0019
n        510,510  486,610  385,119
```
    Runtime on an M1 Pro (16 GB): about 2.5-3 hours of compute for all 51 cells (OLS ~1 min per
    cell, EN and RF ~4-7 min per cell). The run spanned ~50 hours of wall-clock time only because
    the laptop slept for ~45 of them (macOS power log); nothing runs during sleep.

    `config.OOS_START` is now pinned at 2015-01, the start every cell above was fitted with, so
    no refit is needed. `phase2_grid.py` reads it from config and refuses to resume on top of
    cells saved under a different start; `phase2_results.py` checks every cell agrees.

- `bali_replication.py` / `bali_oos.py` (earlier, separate replication of Bali et al.):
    does RF beat OLS one month ahead? Design differs from the main pipeline: T1 = first 36
    months to estimate, T2 = next 24 to tune RF (depth 4, min leaf 50 chosen), then monthly
    refits on all prior months from 2007-07 onward, full panel, raw `retxrf`. Reported:
    OLS +0.31%, RF +0.55% (`results/replication/`). Superseded as the reported replication by
    the in-pipeline version (`phase2_results.py`, section 6), because it validates a
    different design. A full rerun reproduces it to the reported precision (OLS 0.3095% vs
    0.3094%, RF 0.5503% vs 0.5519%, same chosen RF parameters); the RF is not bit-identical,
    likely because of a different scikit-learn version.

    Rerun 2026-11-01 with `requirements-lock.txt`: tuning reproduces OLS to 1e-15 and picks
    the same RF setting, but RF's validation R^2 is -0.184 vs the committed -0.199, so the RF
    numbers are not bit-reproducible on these package versions (likely a scikit-learn version
    difference; the original versions are not recorded). The monthly OOS loop takes hours.

- `phase2_results.py`, section 6 (primary specification, after the Phase 2 proposal feedback):
    are the conclusions robust to scoring every level and horizon on one sample, with L2 forecasts
    demeaned (IC averaged) within issuer-month? The sample common to all 36 main cells is 385,119
    bond-months, 2015-01 to 2021-11 (83 months). Only L2 at h=1 beats a zero forecast: OLS
    +1.60%, EN +1.42%, RF +1.29% (DM p < 0.01 each); no other cell has p < 0.3. No model beats
    another at L2 (all p > 0.3); EN and RF beat OLS only at L0 h=12 (p = 0.021, 0.019). 341 vs 40
    predictors: EN better with all 341 at h=1 (p < 0.001), RF better with the 40 at h=3
    (p = 0.007), otherwise not significant. L2 h=1 is positive for every model in every
    sub-period (2020 weakest, 0.4-0.8%). Issuer-constant predictors give an exactly zero
    demeaned L2 forecast (checked).

    Replication inside the pipeline (OLS/EN/RF, L0, h=1): RF > OLS in every sample, never
    significantly. The sign depends on the test window: on the h=1 sample all models are negative
    (OLS -2.55%, RF -1.92%); cutting that sample at 2021-11 gives OLS -0.20%, RF +0.68%; on the
    primary sample OLS +0.74%, RF +0.89%. Dec 2021 - Oct 2022 holds 23% of the h=1 sample's
    squared L0 returns in 11 months.

- `phase2_results.py`, section 5 (sub-period robustness; periods in `config.SUBPERIODS`,
    scoring in `src/harness.py:score_by_period`):
    does the full-sample pattern hold within pre-COVID (2015-01 to 2019-12), COVID (2020) and
    post-COVID (2021-01 onward) forecast-origin months? Every cell is re-scored from its saved
    predictions, no refit; written to `output/phase2/subperiods.csv` and `tables.md`.

    L2 at h=1 holds: positive in all three periods under RF, and pre- and post-COVID under OLS
    and EN (both ~0 in 2020). The L1/IC h=12 gains do not: EN/RF are negative pre- and
    post-COVID and strongly positive only in 2020 (EN L1 h=12: -0.055 / +0.251 / -0.720). The
    2020 rally and the 2022 sell-off dominate the zero-benchmark denominator; against the
    prevailing training mean the swings shrink sharply (EN L1 h=12: -0.038 / +0.089 / 0.000).
    EN/RF beat OLS on L2 h=1 pre-COVID and in 2020, but OLS is best post-COVID. Within-issuer
    predictors stay within ~0.002 of all 341 at h=1/3 and ~0.01 at h=12 for OLS/EN in every
    period, and beat all 341 under RF post-COVID.

    Caveats: 12 COVID months and 22/20/11 post-COVID months at h=1/3/12, too few for
    per-period DM tests, so sub-period R^2 is read directionally. Periods are by forecast
    origin, so h=12 signals from 2019 already carry 2020 returns. Checks: every cell's
    periods sum to its full-sample row count (51 of 51).

```
OOS R^2 (zero benchmark), L2, all predictors, by forecast-origin period:
                   h=1      h=3      h=12
ols  pre_covid   +0.0063  -0.0185  -0.1208
ols  covid       -0.0008  +0.0089  +0.0534
ols  post_covid  +0.0244  -0.0074  +0.1888
enet pre_covid   +0.0134  +0.0030  -0.0075
enet covid       -0.0002  +0.0046  +0.0017
enet post_covid  +0.0234  +0.0165  +0.1442
rf   pre_covid   +0.0116  -0.0014  -0.0132
rf   covid       +0.0052  +0.0100  +0.0104
rf   post_covid  +0.0138  +0.0083  +0.0690
months           60/12/22 60/12/20 60/12/11   (pre/covid/post)
```
