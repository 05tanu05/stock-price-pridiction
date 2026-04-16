import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json


def fig_to_json(fig):
    return json.loads(fig.to_json())


def build_price_chart(df, pred_result, ticker):
    """Fast price chart with line + forecast. Shows last 90 days only."""
    df_chart = df.tail(90)
    dates = df_chart.index.strftime("%Y-%m-%d").tolist()
    closes = df_chart["Close"].tolist()

    fig = go.Figure()

    # Price line (much faster than candlestick)
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        line=dict(color="#2c1f10", width=2),
        name="Price",
        fill="tozeroy",
        fillcolor="rgba(44,31,16,0.06)"
    ))

    # Future forecast
    if pred_result and pred_result.get("future_prices"):
        last_date = df_chart.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(pred_result["future_prices"]) + 1, freq="B")[1:]
        future_date_strs = future_dates.strftime("%Y-%m-%d").tolist()
        future_prices = pred_result["future_prices"]

        fig.add_trace(go.Scatter(
            x=[dates[-1]] + future_date_strs,
            y=[closes[-1]] + future_prices,
            line=dict(color="#2d6a4f", width=2.5),
            name="30-Day Forecast",
            mode="lines+markers",
            marker=dict(size=4)
        ))

        # Confidence band
        std_est = np.std(closes[-30:]) * 0.5
        upper_band = [p + std_est * (i + 1) ** 0.4 for i, p in enumerate(future_prices)]
        lower_band = [p - std_est * (i + 1) ** 0.4 for i, p in enumerate(future_prices)]

        fig.add_trace(go.Scatter(
            x=future_date_strs + future_date_strs[::-1],
            y=upper_band + lower_band[::-1],
            fill="toself",
            fillcolor="rgba(45,106,79,0.10)",
            line=dict(color="rgba(45,106,79,0)"),
            name="Forecast Range",
        ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Price & AI Forecast", font=dict(size=18, color="#2c1f10")),
        paper_bgcolor="#d6c4ae",
        plot_bgcolor="#cbb99f",
        font=dict(color="#1a1008"),
        legend=dict(bgcolor="rgba(200,180,154,0.8)", bordercolor="rgba(44,31,16,0.25)", borderwidth=1),
        height=460,
        margin=dict(l=50, r=30, t=60, b=30),
        xaxis=dict(gridcolor="rgba(44,31,16,0.10)", color="#6b5744"),
        yaxis=dict(gridcolor="rgba(44,31,16,0.10)", color="#6b5744"),
    )

    return fig_to_json(fig)


def build_profit_loss_chart(current_price, future_prices):
    """Bar chart showing potential profit/loss at different time horizons."""
    horizons = [7, 14, 21, 30]
    labels = ["7 Days", "14 Days", "21 Days", "30 Days"]
    pnl = []
    colors = []
    for h in horizons:
        if h <= len(future_prices):
            target = future_prices[h - 1]
            change_pct = ((target - current_price) / current_price) * 100
            pnl.append(round(change_pct, 2))
            colors.append("#2d6a4f" if change_pct >= 0 else "#c0392b")
        else:
            pnl.append(0)
            colors.append("#888")

    fig = go.Figure(go.Bar(
        x=pnl, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in pnl],
        textposition="outside",
        textfont=dict(color="#2c1f10")
    ))
    fig.update_layout(
        title=dict(text="Predicted Profit / Loss (%)", font=dict(size=14, color="#2c1f10")),
        paper_bgcolor="#d6c4ae",
        plot_bgcolor="#cbb99f",
        font=dict(color="#1a1008"),
        xaxis=dict(gridcolor="rgba(44,31,16,0.10)", ticksuffix="%", color="#6b5744"),
        yaxis=dict(gridcolor="rgba(44,31,16,0.10)", color="#6b5744", autorange="reversed"),
        height=300,
        margin=dict(l=80, r=50, t=50, b=40)
    )
    return fig_to_json(fig)
