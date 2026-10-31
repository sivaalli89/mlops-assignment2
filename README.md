# Feature Store + ML Pipeline (athletes.csv)

## Assignment
**Goal:** design a Feature Store with an ML Pipeline, using `athletes.csv`.

1. Set up an ML pipeline in any MLOps platform of your choice.
2. Use a feature store with the ML pipeline.
3. Load the data and create features with different versions.
4. Run experiments with the ML pipeline and feature store.
5. Create **2 different versions** of the features.
6. Train the **same algorithm** with **2 different sets of hyperparameters** and the 2 feature versions (so **4 combinations**). Do not use AutoML to select the algorithm.
7. Compare the results of the experiments both **quantitatively** (model metrics) and **qualitatively** (model plots).
8. Compare the **carbon emissions** of the different experiments.
9. Submit working code for the above through a GitHub repo.

## Overview
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
mlflow ui --port 5055                      # optional: browse runs at http://localhost:5055
```
`data/athletes.csv` must be present.
(The notebook creates `feature_repo/`, where Feast keeps its config, parquet files and registry.)

## MLflow reference
Every run is tracked in the `athletes-deadlift` experiment. Each run logs the feature version, the
hyperparameters, the metrics (RMSE, MAE, R²) and the carbon emissions (`emissions_kg`).
The screenshot below is the MLflow UI listing the runs (a run name appears more than once because the
notebook was run several times).

![MLflow UI showing the athletes-deadlift runs](images/mlflow_runs.png)

## Run on Databricks
1. Upload `athletes.csv` to `/Volumes/workspace/default/raw/`.
2. Import `databricks_pipeline.ipynb` into your workspace and run all cells.

## Results (local)
| run | RMSE | MAE | R² | emissions (kg CO2) |
|---|---|---|---|---|
| v1_HP-A | 60.70 | 46.10 | 0.610 | 1.4e-06 |
| v1_HP-B | 60.94 | 46.35 | 0.607 | 1.7e-06 |
| v2_HP-A | 36.98 | 27.61 | 0.855 | 1.6e-06 |
| v2_HP-B | 36.86 | 27.49 | 0.856 | 5.3e-06 |

The plots (predicted vs actual, residuals and feature
importance for each run, plus the RMSE and carbon bar charts) are shown in the notebook output.

## Results (Databricks, Spark ML)
| run | RMSE | MAE | R² | emissions (kg CO2) |
|---|---|---|---|---|
| v1_HP-A | 59.64 | 45.76 | 0.614 | 4.0e-05 |
| v1_HP-B | 59.42 | 45.56 | 0.616 | 8.7e-05 |
| v2_HP-A | 37.61 | 27.96 | 0.846 | 2.2e-05 |
| v2_HP-B | 37.40 | 27.80 | 0.848 | 1.0e-04 |

## Analysis
- **Features matter most.** v2 (with the other lifts) cuts the error from about 60 to about 37 in both
  tracks.
- **Hyperparameters matter little.** HP-B (bigger forest) changes the RMSE by less than 0.3 compared
  with HP-A.
- **Bigger forest, more carbon.** HP-B emits about 1.2 to 3.3 times more than HP-A locally and about 2 to
  5 times more on Databricks, for almost no gain. HP-A is the better trade-off.
- **Local vs Databricks.** Results are similar. Databricks errors are slightly higher because Spark ML
  is not identical to scikit-learn, and its emissions are higher because it runs on different hardware.
  CodeCarbon values are estimates, so compare runs within one track.
