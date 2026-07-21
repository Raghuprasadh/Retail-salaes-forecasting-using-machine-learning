"""
eda_utils.py
------------
Reusable Plotly chart builders for the Exploratory Data Analysis page.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def sales_trend_chart(df, date_col, sales_col):
    data = df[[date_col, sales_col]].dropna().sort_values(date_col)
    fig = px.line(data, x=date_col, y=sales_col, title="Sales Trend Over Time")
    fig.update_layout(template="plotly_white", height=450)
    return fig


def monthly_sales_chart(df, date_col, sales_col):
    data = df.copy()
    data["Month"] = pd.to_datetime(data[date_col]).dt.to_period("M").astype(str)
    grouped = data.groupby("Month")[sales_col].sum().reset_index()
    fig = px.bar(grouped, x="Month", y=sales_col, title="Monthly Sales")
    fig.update_layout(template="plotly_white", height=450, xaxis_tickangle=-45)
    return fig


def yearly_sales_chart(df, date_col, sales_col):
    data = df.copy()
    data["Year"] = pd.to_datetime(data[date_col]).dt.year
    grouped = data.groupby("Year")[sales_col].sum().reset_index()
    fig = px.bar(grouped, x="Year", y=sales_col, title="Yearly Sales", text_auto=".2s")
    fig.update_layout(template="plotly_white", height=450)
    return fig


def category_sales_chart(df, category_col, sales_col):
    grouped = df.groupby(category_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
    fig = px.bar(grouped, x=category_col, y=sales_col, title=f"Sales by {category_col}", color=category_col)
    fig.update_layout(template="plotly_white", height=450, showlegend=False)
    return fig


def region_sales_chart(df, region_col, sales_col):
    grouped = df.groupby(region_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
    fig = px.pie(grouped, names=region_col, values=sales_col, title=f"Sales Distribution by {region_col}", hole=0.4)
    fig.update_layout(template="plotly_white", height=450)
    return fig


def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                     title="Correlation Heatmap")
    fig.update_layout(height=550)
    return fig


def histogram_chart(df, column):
    fig = px.histogram(df, x=column, nbins=40, title=f"Distribution of {column}", marginal="box")
    fig.update_layout(template="plotly_white", height=450)
    return fig


def box_plot_chart(df, column, group_col=None):
    if group_col and group_col in df.columns:
        fig = px.box(df, x=group_col, y=column, title=f"Box Plot of {column} by {group_col}", color=group_col)
    else:
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
    fig.update_layout(template="plotly_white", height=450, showlegend=False)
    return fig


def pair_plot_matrix(df, columns):
    fig = px.scatter_matrix(df, dimensions=columns, title="Pair Plot")
    fig.update_layout(height=700)
    return fig


def compute_statistical_insights(df, sales_col):
    return {
        "Total Sales": df[sales_col].sum(),
        "Average Sales": df[sales_col].mean(),
        "Median Sales": df[sales_col].median(),
        "Std Deviation": df[sales_col].std(),
        "Minimum Sales": df[sales_col].min(),
        "Maximum Sales": df[sales_col].max(),
        "Skewness": df[sales_col].skew(),
        "Kurtosis": df[sales_col].kurt(),
    }
