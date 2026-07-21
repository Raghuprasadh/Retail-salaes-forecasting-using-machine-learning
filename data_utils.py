"""
data_utils.py
--------------
Helper functions for dataset validation, cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
import io


def validate_dataset(df: pd.DataFrame):
    """
    Basic validation on an uploaded dataset.
    Returns (is_valid: bool, message: str)
    """
    if df is None or df.empty:
        return False, "The uploaded file is empty."
    if df.shape[1] < 2:
        return False, "Dataset must contain at least two columns."
    if df.shape[0] < 10:
        return False, "Dataset must contain at least 10 rows for meaningful analysis."
    return True, "Dataset looks valid."


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary of high level dataset info."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing.values,
        "Missing %": missing_pct.values
    }).sort_values("Missing Values", ascending=False)

    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    return {
        "shape": df.shape,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": dtypes_df,
        "missing": missing_df,
        "duplicates": int(df.duplicated().sum()),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame, strategy: str, columns=None) -> pd.DataFrame:
    """
    strategy: one of
        'drop_rows'      - drop rows with any missing values
        'mean'           - fill numeric columns with mean
        'median'         - fill numeric columns with median
        'mode'           - fill all columns with mode
        'ffill'          - forward fill
        'bfill'          - backward fill
        'zero'           - fill numeric columns with 0
    """
    df = df.copy()
    cols = columns if columns else df.columns.tolist()

    if strategy == "drop_rows":
        df = df.dropna(subset=cols)
    elif strategy == "mean":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].mean())
    elif strategy == "median":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
    elif strategy == "mode":
        for c in cols:
            if not df[c].mode().empty:
                df[c] = df[c].fillna(df[c].mode()[0])
    elif strategy == "ffill":
        df[cols] = df[cols].ffill()
    elif strategy == "bfill":
        df[cols] = df[cols].bfill()
    elif strategy == "zero":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(0)

    return df.reset_index(drop=True)


def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns]).reset_index(drop=True)


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    return df.rename(columns=rename_map)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------
# Power BI export helpers
# ---------------------------------------------------------------------

def build_powerbi_tables(cleaned_df: pd.DataFrame, date_col: str, target_col: str,
                          category_col=None, region_col=None) -> dict:
    """
    Build a small set of flat, Power-BI-friendly tables (star-schema style)
    from the cleaned dataset:
      - fact_sales: one row per transaction, standardized column names
      - dim_date: one row per calendar date with year/month/quarter/weekday
      - monthly_summary / category_summary / region_summary: pre-aggregated
        tables useful for quick Power BI visuals without extra DAX.
    """
    fact_sales = pd.DataFrame({
        "Date": pd.to_datetime(cleaned_df[date_col]).dt.strftime("%Y-%m-%d"),
        "Sales": cleaned_df[target_col],
    })
    if category_col and category_col in cleaned_df.columns:
        fact_sales["Category"] = cleaned_df[category_col]
    if region_col and region_col in cleaned_df.columns:
        fact_sales["Region"] = cleaned_df[region_col]
    if "Store" in cleaned_df.columns:
        fact_sales["Store"] = cleaned_df["Store"]

    dates = pd.to_datetime(cleaned_df[date_col]).drop_duplicates().sort_values()
    dim_date = pd.DataFrame({
        "Date": dates.dt.strftime("%Y-%m-%d"),
        "Year": dates.dt.year,
        "Month": dates.dt.month,
        "MonthName": dates.dt.strftime("%b"),
        "Quarter": dates.dt.quarter,
        "Weekday": dates.dt.day_name(),
        "IsWeekend": dates.dt.dayofweek.isin([5, 6]),
    })

    monthly_summary = (
        cleaned_df.assign(Month=pd.to_datetime(cleaned_df[date_col]).dt.to_period("M").astype(str))
        .groupby("Month")[target_col].agg(TotalSales="sum", AvgSales="mean", Transactions="count")
        .reset_index()
    )

    tables = {
        "fact_sales": fact_sales,
        "dim_date": dim_date,
        "monthly_summary": monthly_summary,
    }
    if category_col and category_col in cleaned_df.columns:
        tables["category_summary"] = (
            cleaned_df.groupby(category_col)[target_col]
            .agg(TotalSales="sum", AvgSales="mean", Transactions="count")
            .reset_index().rename(columns={category_col: "Category"})
        )
    if region_col and region_col in cleaned_df.columns:
        tables["region_summary"] = (
            cleaned_df.groupby(region_col)[target_col]
            .agg(TotalSales="sum", AvgSales="mean", Transactions="count")
            .reset_index().rename(columns={region_col: "Region"})
        )
    return tables


# ---------------------------------------------------------------------
# Auto-detection helpers (used to run the pipeline without user input)
# ---------------------------------------------------------------------

_DATE_HINTS = ["date", "order_date", "invoice_date", "transaction_date", "day", "timestamp", "period"]
_TARGET_HINTS = ["sales", "revenue", "amount", "total", "price", "value", "units_sold", "quantity"]
_CATEGORY_HINTS = ["category", "product", "department", "segment", "item"]
_REGION_HINTS = ["region", "state", "city", "location", "store", "branch", "country"]


def _best_name_match(columns, hints):
    lower_map = {c: str(c).lower() for c in columns}
    for hint in hints:
        for c, low in lower_map.items():
            if hint == low:
                return c
    for hint in hints:
        for c, low in lower_map.items():
            if hint in low:
                return c
    return None


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """
    Best-effort automatic detection of the date column, the sales/target
    column, and optional category/region columns, so the app can run
    end-to-end without asking the user to map columns manually.
    """
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    object_cols = [c for c in columns if c not in numeric_cols]

    # --- Date column ---
    date_col = _best_name_match(columns, _DATE_HINTS)
    if date_col is None:
        # try parsing each object column and pick the one with the highest parse rate
        best_rate, best_col = 0.0, None
        for c in object_cols:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                rate = parsed.notna().mean()
                if rate > best_rate:
                    best_rate, best_col = rate, c
            except Exception:
                continue
        if best_rate > 0.6:
            date_col = best_col
    if date_col is None and columns:
        date_col = columns[0]

    # --- Target / sales column ---
    target_col = _best_name_match(numeric_cols, _TARGET_HINTS)
    if target_col is None and numeric_cols:
        # fall back to the numeric column (excluding the date column) with the largest variance
        candidates = [c for c in numeric_cols if c != date_col]
        if candidates:
            target_col = df[candidates].var().idxmax()
        else:
            target_col = numeric_cols[0]

    # --- Category / Region columns (optional) ---
    category_col = _best_name_match(object_cols, _CATEGORY_HINTS)
    region_col = _best_name_match(object_cols, _REGION_HINTS)
    if category_col == region_col:
        region_col = None

    return {
        "date_col": date_col,
        "target_col": target_col,
        "category_col": category_col,
        "region_col": region_col,
    }


def auto_clean(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """
    Apply a sensible, fully-automatic cleaning pass:
    remove duplicates, parse the date column, drop rows with an
    unparseable date or missing target, and median/mode-impute the rest.
    """
    df = remove_duplicates(df)
    df = convert_to_datetime(df, date_col)
    df = df.dropna(subset=[date_col, target_col]).reset_index(drop=True)

    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                mode = df[c].mode()
                if not mode.empty:
                    df[c] = df[c].fillna(mode[0])

    return df.sort_values(date_col).reset_index(drop=True)
