"""
feature_engineering.py
-----------------------
Automatic generation of date-based, lag and rolling window features
used by the machine learning models.
"""

import pandas as pd
import numpy as np


def create_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create calendar based features from a datetime column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["Week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Weekday"] = df[date_col].dt.dayofweek
    df["Quarter"] = df[date_col].dt.quarter
    df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)
    return df


def create_lag_features(df: pd.DataFrame, target_col: str, date_col: str, lags=(1, 7, 14, 30)) -> pd.DataFrame:
    """Create lag features on a target column, assuming df is sorted by date."""
    df = df.sort_values(date_col).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, date_col: str, windows=(7, 14, 30)) -> pd.DataFrame:
    """Create rolling mean / std features on a target column."""
    df = df.sort_values(date_col).copy()
    for w in windows:
        df[f"rolling_mean_{w}"] = df[target_col].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df[target_col].shift(1).rolling(window=w).std()
    return df


def build_feature_set(df: pd.DataFrame, date_col: str, target_col: str,
                       lags=(1, 7, 14, 30), windows=(7, 14, 30), dropna=True) -> pd.DataFrame:
    """
    Aggregate the dataset to a daily total (if multiple rows share a date),
    then create date, lag and rolling features. Returns the engineered dataframe.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Aggregate to one row per date so lag/rolling features are meaningful
    daily = df.groupby(date_col, as_index=False)[target_col].sum()
    daily = daily.sort_values(date_col).reset_index(drop=True)

    daily = create_date_features(daily, date_col)
    daily = create_lag_features(daily, target_col, date_col, lags)
    daily = create_rolling_features(daily, target_col, date_col, windows)

    if dropna:
        daily = daily.dropna().reset_index(drop=True)

    return daily
