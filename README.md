# Feature Store + ML Pipeline (athletes.csv)

Predict an athlete's `deadlift` with a `RandomForestRegressor` (chosen by hand, no AutoML).
4 experiments = 2 feature versions x 2 hyperparameter sets, compared on metrics, plots and carbon.

| | Notebook | Feature store | Tracking | Carbon |
|---|---|---|---|---|
| **Local** | `local_pipeline.ipynb` | Feast | MLflow | CodeCarbon |
| **Databricks** | `databricks_pipeline.ipynb` | Databricks Feature Engineering (Unity Catalog) | Managed MLflow | CodeCarbon |

- **Features v1:** age, height, weight, is_male, bmi
- **Features v2:** v1 + candj, snatch, backsq + each lift divided by bodyweight
- **Hyperparameters:** HP-A = 50 trees, depth 6; HP-B = 100 trees, depth 10
  (HP-B is kept small because Spark on Databricks serverless stops training when a model exceeds ~100 MB,
  so the same values are used in both notebooks)
- Each notebook is self-contained (cleaning, feature store, training, plots and comparison are all in it).
  Both use the same cleaning, split (80/20, seed 42) and plots.

## Run locally
```
conda create -n mlops python=3.11
conda activate mlops
pip install -r requirements.txt
jupyter notebook local_pipeline.ipynb      # run all cells from the repo root
mlflow ui                                  # optional: browse runs at http://localhost:5000
```
`data/athletes.csv` must be present. Outputs go to `results/`.
(The notebook creates `feature_repo/`, where Feast keeps its config, parquet files and registry.)

## Run on Databricks
1. Upload `athletes.csv` to `/Volumes/workspace/default/raw/`.
2. Import `databricks_pipeline.ipynb` into your workspace and run all cells.

## Results (local)
| run | RMSE | MAE | R² | emissions (kg CO2) |
|---|---|---|---|---|
| v1_HP-A | 58.32 | 44.88 | 0.632 | 2.0e-06 |
| v1_HP-B | 58.28 | 44.93 | 0.632 | 3.8e-06 |
| v2_HP-A | 36.60 | 27.88 | 0.855 | 3.8e-06 |
| v2_HP-B | 36.04 | 27.47 | 0.859 | 1.0e-05 |

The table is saved to `results/comparison.csv`. The plots (predicted vs actual, residuals and feature
importance for each run, plus the RMSE and carbon bar charts) are shown in the notebook output.

## Analysis
- **Feature version matters most.** Adding the other lifts (v2) cuts RMSE from ~58 to ~37 lb and
  lifts R² from ~0.63 to ~0.86. The feature importance plots show the lifts dominate.
- **Hyperparameters matter little.** On v1 HP-B is tied with HP-A (RMSE 58.28 vs 58.32). On v2 it is
  a little better (36.04 vs 36.60, about 1.5%).
- **Carbon:** HP-B emits about 2 to 3 times more than HP-A (about 2.8x on v2) for that small gain, so
  whether it is worth it depends on how much accuracy matters. The feature version change from v1 to v2
  improves RMSE by far more than any hyperparameter change, at a much smaller carbon cost.
  All runs are tiny in absolute terms (micro-kg CO2).
- **Caveat:** CodeCarbon estimates energy when it cannot read hardware power. The notebooks use the
  grid intensity of the USA, so use the values to compare runs, not as absolute figures.
