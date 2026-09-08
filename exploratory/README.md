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
