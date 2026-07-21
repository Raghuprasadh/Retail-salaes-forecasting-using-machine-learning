"""
sql_utils.py
------------
In-app SQL query layer. Loads the cleaned dataset (and forecast results)
into an in-memory SQLite database so the user can run arbitrary SQL
directly against their data from within the Streamlit app.
"""

import sqlite3
import pandas as pd


def build_in_memory_db(cleaned_df: pd.DataFrame, date_col: str, target_col: str,
                        category_col=None, region_col=None, forecast_df: pd.DataFrame = None) -> sqlite3.Connection:
    """
    Create a fresh in-memory SQLite DB with a 'sales' table (standardized
    column names: sale_date, category, region, sales_amount) and, if
    available, a 'forecast' table (forecast_date, predicted_sales).
    """
    conn = sqlite3.connect(":memory:")

    sales = pd.DataFrame({
        "sale_date": pd.to_datetime(cleaned_df[date_col]).dt.strftime("%Y-%m-%d"),
        "sales_amount": cleaned_df[target_col],
    })
    sales["category"] = cleaned_df[category_col] if category_col and category_col in cleaned_df.columns else None
    sales["region"] = cleaned_df[region_col] if region_col and region_col in cleaned_df.columns else None
    if "Store" in cleaned_df.columns:
        sales["store"] = cleaned_df["Store"]

    sales.to_sql("sales", conn, if_exists="replace", index=False)

    if forecast_df is not None and not forecast_df.empty:
        fc = forecast_df.rename(columns={"Date": "forecast_date", "Forecast": "predicted_sales"}).copy()
        fc["forecast_date"] = pd.to_datetime(fc["forecast_date"]).dt.strftime("%Y-%m-%d")
        fc.to_sql("forecast", conn, if_exists="replace", index=False)

    return conn


# A read-only allowlist keeps the query box safe: only SELECT / WITH
# (CTE) statements are permitted, so the exploratory tool can't be used
# to modify or drop the in-memory tables.
_DISALLOWED_KEYWORDS = ("insert", "update", "delete", "drop", "alter", "create", "attach", "pragma")


def is_safe_select(query: str) -> bool:
    q = query.strip().lower()
    if not (q.startswith("select") or q.startswith("with")):
        return False
    return not any(kw in q for kw in _DISALLOWED_KEYWORDS)


def run_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    if not is_safe_select(query):
        raise ValueError("Only SELECT / WITH (read-only) queries are allowed here.")
    return pd.read_sql_query(query, conn)


PRESET_QUERIES = {
    "Total sales overview": (
        "SELECT COUNT(*) AS transactions, ROUND(SUM(sales_amount),2) AS total_sales, "
        "ROUND(AVG(sales_amount),2) AS avg_sales, ROUND(MIN(sales_amount),2) AS min_sales, "
        "ROUND(MAX(sales_amount),2) AS max_sales FROM sales;"
    ),
    "Monthly sales trend": (
        "SELECT strftime('%Y-%m', sale_date) AS month, ROUND(SUM(sales_amount),2) AS monthly_sales "
        "FROM sales GROUP BY month ORDER BY month;"
    ),
    "Sales by category (ranked)": (
        "SELECT category, ROUND(SUM(sales_amount),2) AS total_sales, "
        "RANK() OVER (ORDER BY SUM(sales_amount) DESC) AS sales_rank "
        "FROM sales WHERE category IS NOT NULL GROUP BY category ORDER BY total_sales DESC;"
    ),
    "Sales by region (% of total)": (
        "SELECT region, ROUND(SUM(sales_amount),2) AS total_sales, "
        "ROUND(100.0*SUM(sales_amount)/(SELECT SUM(sales_amount) FROM sales),2) AS pct_of_total "
        "FROM sales WHERE region IS NOT NULL GROUP BY region ORDER BY total_sales DESC;"
    ),
    "7-day moving average": (
        "SELECT sale_date, ROUND(daily_sales,2) AS daily_sales, "
        "ROUND(AVG(daily_sales) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),2) AS moving_avg_7d "
        "FROM (SELECT sale_date, SUM(sales_amount) AS daily_sales FROM sales GROUP BY sale_date) ORDER BY sale_date;"
    ),
    "Month-over-month growth %": (
        "WITH monthly AS (SELECT strftime('%Y-%m', sale_date) AS month, SUM(sales_amount) AS total_sales "
        "FROM sales GROUP BY month) "
        "SELECT month, ROUND(total_sales,2) AS total_sales, "
        "ROUND(100.0*(total_sales - LAG(total_sales) OVER (ORDER BY month))/LAG(total_sales) OVER (ORDER BY month),2) AS mom_growth_pct "
        "FROM monthly ORDER BY month;"
    ),
    "Forecast vs latest actuals": (
        "SELECT * FROM forecast ORDER BY forecast_date;"
    ),
}
