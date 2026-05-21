from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm
from scipy.optimize import minimize

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Paths
PROCESSED = Path("data/processed")
OUTPUT = Path("output")
OUTPUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = PROCESSED / "gkx_master_panel.parquet"

# OLS3 Model Setup
OLS3_RAW_COLS = ["mvel1", "bm", "mom12m"]

# After rank-transforming, these inputs are used
OLS_Feature_Cols = [f"{c}_rank" for c in OLS3_RAW_COLS]
Model_Name = "OLS3_Huber"

# GKX has testing period 1987-2016
Test_Years = range(1987, 2017)

# Using next-month excess return
# characteristics at month t -> return at month t+1
# Dont know if they are pre-aligned, so keeping option for both
Target_Mode = "next_month"

# GKX sample starts from 1957
Start_Target_Month_Id = 1957 * 12 + 3
End_Target_Month_Id = 2016 * 12 + 12

# Helper Functions
# GKX style out-of-sample R2
# The denominator is the sum of squared realised returns
# Benchmarks against a zero forecast, not a historical mean forecast
def oos_r2(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)

# Fitting linear regression with a GKX style Huber Loss
# Huber loss:
#        e²                       if |e| <= xi
#        2 * xi * |e| - xi²       if |e| > xi

# xi is set as the 99.9% quantile of |y| in the training sample.
# This is a practical implementation of the GKX +H idea.
def fit_huber_linear(X: np.ndarray, y: np.ndarray, xi_quantile: float = 0.999):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    if X.shape[0] == 0:
        raise ValueError("No valid observations after removing missing values.")

    # intercept
    X_aug = np.column_stack([np.ones(X.shape[0]), X])

    # Huber threshold
    xi = np.quantile(np.abs(y), xi_quantile)

    if xi <= 0 or not np.isfinite(xi):
        xi = 1e-8

    # Initial guess: ordinary least squares
    beta0 = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    def objective_and_gradient(beta):
        residual = y - X_aug @ beta
        abs_residual = np.abs(residual)

        loss_vec = np.where(
            abs_residual <= xi,
            residual ** 2,
            2 * xi * abs_residual - xi ** 2
        )

        loss = loss_vec.mean()

        # Derivative of Huber loss with respect to residual,
        # divided by 2 for compact gradient expression.
        psi = np.where(
            abs_residual <= xi,
            residual,
            xi * np.sign(residual)
        )

        gradient = -2.0 * (X_aug.T @ psi) / len(y)

        return loss, gradient

    result = minimize(
        fun=lambda b: objective_and_gradient(b),
        x0=beta0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    if not result.success:
        print(f"Warning: Huber optimization did not fully converge: {result.message}")

    return result.x

# Predict using fitted Huber linear regression coefficients
def predict_huber_linear(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    return X_aug @ beta

# Loads master panel, creates target variable, fills missing characteristics,
# and rank-transforms OLS-3 characteristics into [-1, 1]
# Prepare modelling panel
def prepare_ols3_panel(TARGET_MODE):
    lf = pl.scan_parquet(PANEL_PATH)

    lf = lf.with_columns([
        pl.col("month_id").cast(pl.Int64),
        pl.col("month").dt.year().alias("info_year"),
    ])

    # Creating prediction target
    if TARGET_MODE == "same_month":
        lf = lf.with_columns([
            pl.col("ret_excess").alias("y"),
            pl.col("month").alias("target_month"),
            pl.col("month_id").alias("target_month_id"),
            pl.col("month").dt.year().alias("target_year"),
            pl.col("mktcap_lag").alias("size_for_ranking"),
        ])

    elif TARGET_MODE == "next_month":
        # Attach return from month t+1 to predictors from month t.
        y_lf = lf.select([
            "permno",
            (pl.col("month_id") - 1).alias("month_id"),
            pl.col("ret_excess").alias("y"),
            pl.col("month").alias("target_month"),
            pl.col("month_id").alias("target_month_id"),
            pl.col("month").dt.year().alias("target_year"),
        ])

        lf = lf.join(y_lf, on=["permno", "month_id"], how="left")

        # For target t+1, market cap at information month t is known.
        lf = lf.with_columns(
            pl.col("mktcap").alias("size_for_ranking")
        )

    else:
        raise ValueError("TARGET_MODE must be either 'same_month' or 'next_month'.")

    # Keep target sample
    lf = lf.filter(
        (pl.col("target_month_id") >= Start_Target_Month_Id) &
        (pl.col("target_month_id") <= End_Target_Month_Id) &
        (pl.col("y").is_not_null())
    )

    # Fill missing OLS-3 characteristics
    for c in OLS3_RAW_COLS:
        lf = lf.with_columns(
            pl.col(c)
              .fill_null(pl.col(c).median().over("month_id"))
              .fill_null(0.0)
              .cast(pl.Float64)
              .alias(f"{c}_filled")
        )

    # Cross sectional rank transform into [-1,1]
    lf = lf.with_columns(
        pl.len().over("month_id").alias("_n_month")
    )

    for c in OLS3_RAW_COLS:
        filled_col = f"{c}_filled"
        rank_col = f"{c}_rank"

        lf = lf.with_columns(
            (
                2.0
                * (pl.col(filled_col).rank("average").over("month_id") - 1.0)
                / (pl.col("_n_month") - 1.0)
                - 1.0
            )
            .fill_nan(0.0)
            .cast(pl.Float64)
            .alias(rank_col)
        )

    return lf

def collect_xy(lf: pl.LazyFrame,
    feature_cols: list[str],
    train_max_year: int | None = None,
    test_year: int | None = None):

    q = lf

    if train_max_year is not None:
        q = q.filter(pl.col("target_year") <= train_max_year)

    if test_year is not None:
        q = q.filter(pl.col("target_year") == test_year)

    keep_cols = [
        "permno",
        "month",
        "target_month",
        "month_id",
        "target_month_id",
        "target_year",
        "y",
        "size_for_ranking",
    ] + feature_cols

    df = q.select(keep_cols).collect()

    ids = df.select([
        "permno",
        "month",
        "target_month",
        "month_id",
        "target_month_id",
        "target_year",
        "y",
        "size_for_ranking",
    ])

    X = df.select(feature_cols).to_numpy()
    y = df.get_column("y").to_numpy()

    return ids, X, y

# Running OLS3+H Replication
def run_ols3_huber():
    lf = prepare_ols3_panel(Target_Mode)

    all_predictions = []

    for test_year in tqdm(Test_Years, desc=Model_Name):
        # GKX recursive split:
        # test year 1987 uses training up to 1974 and validation 1975–1986.
        # OLS-3+H has no hyperparameter grid, so validation is not used here.
        train_end_year = test_year - 13

        ids_train, X_train, y_train = collect_xy(
            lf,
            OLS_Feature_Cols,
            train_max_year=train_end_year
        )

        ids_test, X_test, y_test = collect_xy(
            lf,
            OLS_Feature_Cols,
            test_year=test_year
        )

        beta = fit_huber_linear(X_train, y_train, xi_quantile=0.999)
        pred = predict_huber_linear(beta, X_test)

        out = ids_test.with_columns([
            pl.Series("prediction", pred),
            pl.lit(Model_Name).alias("model"),
            pl.lit(train_end_year).alias("train_end_year"),
        ])

        all_predictions.append(out)

    preds = pl.concat(all_predictions)

    pred_path = OUTPUT / f"predictions_{Model_Name}_{Target_Mode}.parquet"
    preds.write_parquet(pred_path, compression="zstd")

    r2_all = oos_r2(
        preds.get_column("y").to_numpy(),
        preds.get_column("prediction").to_numpy()
    )

    # Top-1000 and bottom-1000 by market value
    preds_ranked = (
        preds
        .filter(pl.col("size_for_ranking").is_not_null())
        .with_columns([
            pl.col("size_for_ranking")
              .rank("ordinal", descending=True)
              .over("target_month_id")
              .alias("rank_big"),

            pl.col("size_for_ranking")
              .rank("ordinal", descending=False)
              .over("target_month_id")
              .alias("rank_small"),
        ])
    )

    big = preds_ranked.filter(pl.col("rank_big") <= 1000)
    small = preds_ranked.filter(pl.col("rank_small") <= 1000)

    r2_big = oos_r2(
        big.get_column("y").to_numpy(),
        big.get_column("prediction").to_numpy()
    )

    r2_small = oos_r2(
        small.get_column("y").to_numpy(),
        small.get_column("prediction").to_numpy()
    )

    summary = pl.DataFrame({
        "model": [Model_Name],
        "target_mode": [Target_Mode],
        "r2_all_pct": [100 * r2_all],
        "r2_top1000_pct": [100 * r2_big],
        "r2_bottom1000_pct": [100 * r2_small],
        "n_obs": [preds.height],
        "prediction_file": [str(pred_path)],
    })

    print("\nR² summary:")
    print(summary)

    summary_path = OUTPUT / f"r2_summary_{Model_Name}_{Target_Mode}.csv"
    summary.write_csv(summary_path)

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    run_ols3_huber()


