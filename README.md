# F1 Race Outcome Prediction (2025 Season)

Predicts Formula 1 finishing order using qualifying results, per-circuit domain
features, and rolling driver form  built on data pulled live from the
[FastF1](https://github.com/theOehrly/Fast-F1) API.

## What it does

For each round of the 2025 season, the model predicts every driver's finishing
position using qualifying results and historical form, then ranks those
predictions into a final 1–20 ordering.

## Data pipeline

`main.py` pulls, per round:
- **Qualifying session** → grid position
- **Race session** → classified finishing position, points, status, lap times, sector times
- **Race weather data** → whether rain was recorded during the session

Results are cached locally via `fastf1.Cache`, so the first run downloads from
FastF1's API and every run after that is fast. The merged dataset is saved to
`f1_combined_dataset.csv` so the modeling step doesn't require re-fetching.

## Engineered features

| Feature | What it is | Available pre-race? |
|---|---|---|
| `GridPosition` | Qualifying result | Yes |
| `Track_Downforce` | Hand-coded 1–3 scale per circuit (low/medium/high downforce demand) | Yes |
| `Overtaking_Difficulty` | Hand-coded 1–10 scale per circuit, based on track architecture | Yes |
| `Grid_Penalty_Score` | `GridPosition x Overtaking_Difficulty` — a bad qualifying result matters more on tracks where passing is hard | Yes |
| `Podiums_Last_3_Races` | Rolling count of podiums over a driver's previous 3 races, computed with `shift(1)` so the current race's own result can't leak in | Yes |
| `HasStreak` | Binary flag derived from the above | Yes |
| `IsWetRace` | Whether rain was recorded during *this* race | No — see limitations |
| `Adjusted_PaceDelta` | Median race-pace gap to the fastest driver, scaled by a hand-built per-driver wet-weather skill index | No — see limitations |

## Model

`XGBRegressor` trained to predict `ClassifiedPosition` directly (DNFs coded as
20th). Predictions are converted to a final ranked order via `.rank()`. Evaluated
with mean absolute error in finishing positions on the held-out final round.

## Results

![Actual vs predicted finishing position](results/predicted_vs_actual_r24.png)



MAE for the held-out round is printed at runtime. Performance is strongest at
the front of the grid and noticeably worse for midfield/backmarker positions —
see Known Limitations for why this single-race evaluation should be read with
caution.

## Known limitations and honest next steps

This is a working v1, not a finished system. In order of priority:

1. **`Adjusted_PaceDelta` and `IsWetRace` are computed from the same race
   they're predicting.** Both come from that race's own lap times and weather
   readings, which means they aren't actually known before the race happens.
   As built, this model can be run retroactively on a finished race but not
   prospectively on an upcoming one. Next step: replace `Adjusted_PaceDelta`
   with a rolling average of *prior* races' pace, the same way
   `Podiums_Last_3_Races` already avoids this exact problem.
2. **Trained on a single season.** ~480 rows across 24 races, with only one
   race held out for evaluation. FastF1 has reliable data back to 2018;
   extending `years_to_load` to cover 2018–2025 is a small code change and
   would give a far more trustworthy backtest. Note: the hand-coded circuit
   dictionaries will need extending too, since the pre-2022 calendar includes
   circuits not currently in `TRACK_OVERTAKING_DIFFICULTY` /
   `TRACK_DOWNFORCE_LEVEL`.
3. **No baseline comparison.** The model hasn't been checked against the
   trivial heuristic of "predicted order = grid order." Adding this is cheap
   and is the right sanity check for whether the model adds real signal.
4. **Single-race evaluation.** Once multi-season data is in place, this should
   move to a backtest across many held-out races rather than one.

## Tech stack

Python, FastF1, pandas, numpy, scikit-learn, XGBoost, matplotlib.

## Running it

```bash
pip install fastf1 numpy pandas scikit-learn matplotlib xgboost
python main.py
```

First run pulls data from FastF1 and may take a while; later runs use the
local cache and skip straight to modeling if `f1_combined_dataset.csv`
already exists.