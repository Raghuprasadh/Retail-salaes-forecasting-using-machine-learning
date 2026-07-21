"""
models.py
---------
Training, evaluation and forecasting logic for both regression style
ML models (Linear Regression, Random Forest, XGBoost) and classical
time series models (ARIMA, SARIMA).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ---------------------------------------------------------------------
# ML Regression Models (feature based)
# ---------------------------------------------------------------------

def get_ml_model(name: str):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    if name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        return XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )
    raise ValueError(f"Unknown model: {name}")


def train_ml_model(model_name, X_train, y_train, X_test, y_test):
    model = get_ml_model(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)
    return model, preds, metrics


def feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        return imp
    if hasattr(model, "coef_"):
        imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(model.coef_)
        }).sort_values("Importance", ascending=False)
        return imp
    return None


# ---------------------------------------------------------------------
# Time Series Models (ARIMA / SARIMA)
# ---------------------------------------------------------------------

def train_arima(train_series, test_len, order=(5, 1, 0)):
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=test_len)
    return fitted, forecast


def train_sarima(train_series, test_len, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=test_len)
    return fitted, forecast


def forecast_future_arima(fitted_model, steps):
    return fitted_model.forecast(steps=steps)


# ---------------------------------------------------------------------
# Recursive multi-step forecasting for feature based ML models
# ---------------------------------------------------------------------

def recursive_ml_forecast(model, last_known_df, date_col, target_col, feature_cols,
                           steps, lags=(1, 7, 14, 30), windows=(7, 14, 30)):
    """
    Iteratively forecast `steps` days ahead using a trained feature-based
    ML model, regenerating lag / rolling / date features at each step.
    """
    history = last_known_df[[date_col, target_col]].copy()
    history[date_col] = pd.to_datetime(history[date_col])
    history = history.sort_values(date_col).reset_index(drop=True)

    future_rows = []
    last_date = history[date_col].max()

    for i in range(1, steps + 1):
        next_date = last_date + pd.Timedelta(days=i)

        temp = history.copy()
        row = {date_col: next_date}
        row_df = pd.DataFrame([row])
        row_df[date_col] = pd.to_datetime(row_df[date_col])

        # date features
        row_df["Year"] = row_df[date_col].dt.year
        row_df["Month"] = row_df[date_col].dt.month
        row_df["Day"] = row_df[date_col].dt.day
        row_df["Week"] = row_df[date_col].dt.isocalendar().week.astype(int)
        row_df["Weekday"] = row_df[date_col].dt.dayofweek
        row_df["Quarter"] = row_df[date_col].dt.quarter
        row_df["IsWeekend"] = row_df["Weekday"].isin([5, 6]).astype(int)

        # lag features from history (+ previously forecasted values)
        combined_target = pd.concat([temp[target_col], pd.Series([r["__pred__"] for r in future_rows])], ignore_index=True)
        for lag in lags:
            col = f"lag_{lag}"
            if col in feature_cols:
                row_df[col] = combined_target.iloc[-lag] if len(combined_target) >= lag else np.nan

        # rolling features
        for w in windows:
            mcol, scol = f"rolling_mean_{w}", f"rolling_std_{w}"
            window_vals = combined_target.tail(w)
            if mcol in feature_cols:
                row_df[mcol] = window_vals.mean()
            if scol in feature_cols:
                row_df[scol] = window_vals.std()

        X_next = row_df.reindex(columns=feature_cols, fill_value=0).ffill(axis=0).fillna(0)
        pred = model.predict(X_next)[0]

        future_rows.append({"__date__": next_date, "__pred__": pred})

    forecast_df = pd.DataFrame({
        date_col: [r["__date__"] for r in future_rows],
        "Forecast": [r["__pred__"] for r in future_rows]
    })
    return forecast_df
