# exploratory/

One line per script: what it asked, what it found.

- `key_choice_exploration.py`:
    is `gvkey` or `issuer_cusip` the right issuer key for L2?
    *Pending findings. For now, downstream functions are parameterised to take in either `gvkey` or `issuer_cusip`.*

- `splits_prototype.py`:
    does an `h`-month embargo remove the overlap between a fold's
    training labels and its test labels? Yes, exactly. `s <= T-1` leaks `h-1` calendar months
    per fold; `s <= T-h` leaks none, costing `h` unusable signal months at the panel tail and
    `h-1` training months at the head. Promoted to `src/splits.py`.

    Label convention (`t` is labelled by returns over `t+1 … t+h`) checked against
    `dnr_ml_predictions/README.txt`, where `realized_return_date` is `t+1` for `signal_date`
    `t`. The leakage checks move with it and so cannot catch it; pinned in `test/test_splits.py`.

    Counts below assume `oos_start=2015-01` and are provisional — `config.OOS_START` is unset.

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

    Counts assume `oos_start=2015-01` and are provisional; `config.OOS_START` is unset.

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
