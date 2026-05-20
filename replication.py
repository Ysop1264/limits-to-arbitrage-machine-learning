# Numerical and dataset packages
import numpy as np
import polars as pl
import pyarrow as pa
from pathlib import Path

# Progress Bars
from tqdm import tqdm

# Models
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# Inspection
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(80)

RAW = Path("data/raw")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
OUTPUT = Path("data/output")

for folder in [INTERIM, PROCESSED, OUTPUT]:
    folder.mkdir(parents=True, exist_ok=True)

files = {
    "author": RAW / "author_dataset.csv",
    "wrds": RAW / "wrds_dataset.csv",
    "goyal": RAW / "goyal_welch.csv",
    "ff3mom": RAW / "FF3+Mom_RF.csv",
    "ff5mom": RAW / "FF5+Mom_RF.csv",
}

# Inspection to check for data types to ensure completeness
for name, path in files.items():
    print(f"\n{'=' * 80}")
    print(f"{name.upper()}: {path}")
    print(f"{'=' * 80}")

    lf = pl.scan_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "NULL", "."]
    )

    print("\nSchema:")
    print(lf.collect_schema())

    print("\nHead:")
    print(lf.head(5).collect())

    print("\nRows:")
    print(lf.select(pl.len().alias("n_rows")).collect())

    for name, path in files.items():
        out = INTERIM / f"{name}.parquet"

        if out.exists():
            print(f"{out} already exists. Skipping.")
        continue

        print(f"Converting {name} to Parquet...")

        lf = pl.scan_csv(
            path,
            infer_schema_length=10000,
            null_values=["", "NA", "NaN", "NULL", "."]
        )

        lf.sink_parquet(out, compression="zstd")

print("CSV to Parquet conversion complete.")

# Storing all column names (from author dataset)
PROCESSED.mkdir(parents=True, exist_ok=True)

CHAR_COLS = [
    "absacc", "acc", "aeavol", "age", "agr",
    "baspread", "beta", "betasq", "bm", "bm_ia",
    "cash", "cashdebt", "cashpr", "cfp", "cfp_ia",
    "chatoia", "chcsho", "chempia", "chinv", "chmom",
    "chpmia", "chtx", "cinvest", "convind", "currat",
    "depr", "divi", "divo", "dolvol", "dy",
    "ear", "egr", "ep", "gma", "grcapx",
    "grltnoa", "herf", "hire", "idiovol", "ill",
    "indmom", "invest", "lev", "lgr", "maxret",
    "mom12m", "mom1m", "mom36m", "mom6m", "ms",
    "mvel1", "mve_ia", "nincr", "operprof", "orgcap",
    "pchcapx_ia", "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick",
    "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "pctacc",
    "pricedelay", "ps", "quick", "rd", "rd_mve",
    "rd_sale", "realestate", "retvol", "roaq", "roavol",
    "roeq", "roic", "rsup", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin",
    "sp", "std_dolvol", "std_turn", "stdacc", "stdcf",
    "tang", "tb", "turn", "zerotrade"
]

# AUTHOR DATSET: GKX Characteristics
# Creating a lazy polars datatset for the gkx characteristics
# We use the scan_csv to read the file for data types and them standardisation without
# overloading the memory
# with_columns is used to add or replace columns
# cast is used for amending the data to standarise for later merging
author = (
    # Preparing the csv to be read without completely loading it
    pl.scan_csv(
        RAW / "author_dataset.csv", 
        infer_schema_length=10000, # only use first 10000 rows to guess column type
        null_values=["", "NA", "NaN", "NULL", "null", "."] 
    )
    .with_columns([ 
        pl.col("permno").cast(pl.Int64),
        # Convert data to int, then convert to string, parse the string as date, then rename
        # the column to month
        pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("month"),
        # Convert industry code to int, strict=False ensures that if something 
        # cannot be converted, it is stored as null instead of crashing the program
        pl.col("sic2").cast(pl.Int64, strict=False).alias("sic2"),
    ])
    .with_columns(
        # Constructing a unique month identifier as different datasets have 
        # dates in different formats
        (pl.col("month").dt.year() * 12 + pl.col("month").dt.month()).alias("month_id")
    )
    .with_columns([
        # Converting all GKX characterisitcs to numbers by looping over 
        # the char_cols array
        pl.col(c).cast(pl.Float64, strict=False).alias(c)
        for c in CHAR_COLS
    ])
    .select(["permno", "month", "month_id", "sic2"] + CHAR_COLS)
)

# WRDS CRSP: returns, price, shares, market cap
# Reading the csv lazily using scan_csv to avoid overloading memory
# Renaming columns to match up with other datatsets for future merging
# Renaming data to month, and all characteristics to lowercase 
wrds = (
    pl.scan_csv(
        RAW / "wrds_dataset.csv",
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "NULL", "null", "."],
        schema_overrides={
            # CRSP return fields can contain letter codes like C, S, etc.
            # Read them as strings first, then safely cast to float below.
            "RET": pl.Utf8,
            "DLRET": pl.Utf8,
            "DLPRC": pl.Utf8,

            # These numeric-looking fields may also contain special codes.
            "PRC": pl.Utf8,
            "VOL": pl.Utf8,
            "SHROUT": pl.Utf8,

            # These classification fields can contain non-numeric CRSP codes such as Z.
            "SHRCD": pl.Utf8,
            "EXCHCD": pl.Utf8,
            "SICCD": pl.Utf8,
        }
    )
    .rename({"PERMNO": "permno"})
    .with_columns([
        pl.col("date").str.strptime(pl.Date, "%d/%m/%Y").alias("month"),

        # Convert CRSP return and price variables to numeric.
        # strict=False ensures that if something cannot be converted,
        # it is stored as null instead of crashing the program.
        pl.col("RET").cast(pl.Float64, strict=False).alias("ret"),
        pl.col("DLRET").cast(pl.Float64, strict=False).alias("dlret"),
        pl.col("PRC").cast(pl.Float64, strict=False).alias("prc"),
        pl.col("SHROUT").cast(pl.Float64, strict=False).alias("shrout"),
        pl.col("VOL").cast(pl.Float64, strict=False).alias("vol"),

        # Convert CRSP classification variables safely.
        # Non-numeric values such as Z become null.
        pl.col("SHRCD").cast(pl.Int64, strict=False).alias("shrcd"),
        pl.col("EXCHCD").cast(pl.Int64, strict=False).alias("exchcd"),
        pl.col("SICCD").cast(pl.Int64, strict=False).alias("siccd_crsp"),
    ])
    .with_columns(
        # Unique month identifier
        (pl.col("month").dt.year() * 12 + pl.col("month").dt.month()).alias("month_id")
    )
    .with_columns([
        # CRSP can be negative, but they still represent economic value
        pl.col("prc").abs().alias("price_abs"),

        # Constructing market cap
        (pl.col("prc").abs() * pl.col("shrout")).alias("mktcap"),
    ])
    .with_columns(
        # Adjusting returns for delisting returns
        pl.when(pl.col("ret").is_not_null() & pl.col("dlret").is_not_null())
          .then((1 + pl.col("ret")) * (1 + pl.col("dlret")) - 1)
          .when(pl.col("ret").is_null() & pl.col("dlret").is_not_null())
          .then(pl.col("dlret"))
          .otherwise(pl.col("ret"))
          .alias("ret_adj")
    )
    # Sort by stock and month
    .sort(["permno", "month_id"])
    .with_columns(
        # Lagging market cap
        pl.col("mktcap").shift(1).over("permno").alias("mktcap_lag")
    )
    # Selecting relevant variables
    .select([
        "permno", "month_id", "ret", "dlret", "ret_adj",
        "price_abs", "mktcap", "mktcap_lag",
        "vol", "shrcd", "exchcd", "siccd_crsp"
    ])
)

# FF3 + MOM: risk-free rate
ff3 = (
    pl.scan_csv(
        RAW / "FF3+Mom_RF.csv",
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "NULL", "null", "."]
    )
    .with_columns(
        # Renaming to month to standardise
        pl.col("dateff").str.strptime(pl.Date, "%d/%m/%Y").alias("month")
    )
    .with_columns(
        # Unique month identifier
        (pl.col("month").dt.year() * 12 + pl.col("month").dt.month()).alias("month_id")
    )
    .select([
        "month_id",
        pl.col("rf").cast(pl.Float64).alias("rf")
    ])
)

# GOYAL-WELCH: macro predictors
goyal = (
    pl.scan_csv(
        RAW / "goyal_welch.csv",
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "NULL", "null", "."]
    )
    .with_columns([
        # Isolating year
        (pl.col("yyyymm") // 100).alias("year"),
        # Isolating month
        (pl.col("yyyymm") % 100).alias("month_num"),
        # Casting to float
        pl.col("Index").cast(pl.Float64, strict=False).alias("Index_num"),
        pl.col("D12").cast(pl.Float64, strict=False),
        pl.col("E12").cast(pl.Float64, strict=False),
        pl.col("b/m").cast(pl.Float64, strict=False).alias("bm_macro_raw"),
        pl.col("tbl").cast(pl.Float64, strict=False),
        pl.col("AAA").cast(pl.Float64, strict=False),
        pl.col("BAA").cast(pl.Float64, strict=False),
        pl.col("lty").cast(pl.Float64, strict=False),
        pl.col("ntis").cast(pl.Float64, strict=False),
        pl.col("svar").cast(pl.Float64, strict=False),
    ])
    .with_columns(
        # Unique month identifier
        (pl.col("year") * 12 + pl.col("month_num")).alias("month_id")
    )
    .with_columns([
        # Renaming columns for easier understanding
        (pl.col("D12") / pl.col("Index_num")).log().alias("macro_dp"),
        (pl.col("E12") / pl.col("Index_num")).log().alias("macro_ep"),
        pl.col("bm_macro_raw").alias("macro_bm"),
        pl.col("ntis").alias("macro_ntis"),
        pl.col("tbl").alias("macro_tbl"),
        (pl.col("lty") - pl.col("tbl")).alias("macro_tms"),
        (pl.col("BAA") - pl.col("AAA")).alias("macro_dfy"),
        pl.col("svar").alias("macro_svar"),
    ])
    .select([
        "month_id",
        "macro_dp", "macro_ep", "macro_bm", "macro_ntis",
        "macro_tbl", "macro_tms", "macro_dfy", "macro_svar"
    ])
)

# Merging everything
panel = (
    author
    .join(wrds, on=["permno", "month_id"], how="inner")
    .join(ff3, on="month_id", how="left")
    .join(goyal, on="month_id", how="left")
    .with_columns(
        (pl.col("ret_adj") - pl.col("rf")).alias("ret_excess")
    )
    .filter(
        (pl.col("month").dt.year() >= 1957) &
        (pl.col("month").dt.year() <= 2016)
    )
)

panel.sink_parquet(PROCESSED / "gkx_master_panel.parquet", compression="zstd")
print("Saved data/processed/gkx_master_panel.parquet")

# Inspecting the master panel
master = pl.scan_parquet("data/processed/gkx_master_panel.parquet")

print(
    master.select(
        pl.len().alias("n_rows"),
        pl.col("permno").n_unique().alias("n_permnos"),
        pl.col("month").min().alias("min_month"),
        pl.col("month").max().alias("max_month"),
        pl.col("ret_excess").null_count().alias("missing_ret_excess"),
        pl.col("mktcap_lag").null_count().alias("missing_mktcap_lag")
    ).collect()
)

print(
    master
    .with_columns(pl.col("month").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("permno").n_unique().alias("n_stocks")
    )
    .sort("year")
    .collect()
)
