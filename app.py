import os
import sys
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Preload all modules at startup so first request is fast
print("Loading modules...")
from utils.data_fetcher import fetch_all_data, get_current_info
from utils.indicators import analyze_safety, get_buying_platforms
from utils.charts import build_price_chart, build_profit_loss_chart
from model.lstm_model import train_and_predict
from utils.db import save_search, save_prediction, get_search_history, get_recent_predictions, get_dashboard_stats
print("Ready!")

app = Flask(__name__)
app.secret_key = "stock_prediction_secret_2024"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    query = request.form.get("stock_query", "").strip()
    if not query:
        return render_template("index.html", error="Please enter a stock name or ticker symbol.")

    try:
        # 1. Fetch everything (fetch_all_data retries internally on transient errors)
        ticker_symbol, company_name, df_raw, info = fetch_all_data(query, period="1y")

        if ticker_symbol is None or df_raw is None or df_raw.empty:
            return render_template("index.html",
                error=f"Could not find stock '{query}'. Try using the ticker symbol (e.g., AAPL, GOOGL, TSLA).")

        # 2. Safety analysis
        safety_score, safety_label, safety_color, signals = analyze_safety(df_raw, info)

        # 3. Prediction
        pred_result = train_and_predict(df_raw, forecast_days=30, ticker_symbol=ticker_symbol)

        # 4. Build charts
        price_chart = build_price_chart(df_raw, pred_result, ticker_symbol)

        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or df_raw["Close"].iloc[-1]
        future_prices = pred_result.get("future_prices", [])
        pnl_chart = build_profit_loss_chart(current_price, future_prices)

        # 5. Buying platforms (ranked by stock characteristics)
        # Compute volatility for platform ranking
        _closes = df_raw["Close"].values
        _vol_pct = 2.0
        if len(_closes) >= 30:
            _daily_ret = np.diff(_closes[-30:]) / _closes[-30:-1]
            _vol_pct = float(np.std(_daily_ret) * 100)

        platforms = get_buying_platforms(
            ticker_symbol,
            current_price=current_price,
            safety_score=safety_score,
            volatility_pct=_vol_pct,
            pe_ratio=info.get("trailingPE") or info.get("forwardPE") or 0,
            recommendation=(info.get("recommendationKey") or ""),
        )

        # 6. Current info
        current_info = get_current_info(info)

        # 7. Profit/Loss summary
        pnl_summary = []
        for days in [7, 14, 21, 30]:
            if days <= len(future_prices):
                target = future_prices[days - 1]
                change = target - current_price
                change_pct = (change / current_price) * 100
                pnl_summary.append({
                    "horizon": f"{days} Days",
                    "price": round(target, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "direction": "up" if change >= 0 else "down"
                })

        # 8. 52-week stats
        week52_high = info.get("fiftyTwoWeekHigh", df_raw["High"].max())
        week52_low = info.get("fiftyTwoWeekLow", df_raw["Low"].min())
        prev_close = info.get("previousClose", df_raw["Close"].iloc[-2] if len(df_raw) > 1 else current_price)
        day_change = current_price - prev_close
        day_change_pct = (day_change / prev_close) * 100 if prev_close else 0

        # ── Save to MongoDB ──
        try:
            save_search(ticker_symbol, company_name or ticker_symbol, query)
            save_prediction(
                ticker=ticker_symbol,
                company_name=company_name or ticker_symbol,
                current_price=round(current_price, 2),
                currency=current_info.get("currency", "USD"),
                safety_score=safety_score,
                safety_label=safety_label,
                pnl_summary=pnl_summary,
                model_used=pred_result.get("model_used", "AI Prediction"),
                pred_mape=round(pred_result.get("mape", 0), 2),
            )
        except Exception:
            pass  # Don't break predictions if DB is down

        return render_template(
            "result.html",
            ticker=ticker_symbol,
            company_name=company_name or ticker_symbol,
            query=query,
            current_price=round(current_price, 2),
            currency=current_info.get("currency", "USD"),
            day_change=round(day_change, 2),
            day_change_pct=round(day_change_pct, 2),
            week52_high=round(float(week52_high), 2),
            week52_low=round(float(week52_low), 2),
            market_cap=current_info.get("marketCap"),
            volume=current_info.get("volume"),
            pe_ratio=round(current_info.get("trailingPE", 0) or 0, 2),
            beta=round(current_info.get("beta", 0) or 0, 2),
            sector=current_info.get("sector", "N/A"),
            industry=current_info.get("industry", "N/A"),
            exchange=current_info.get("exchange", "N/A"),
            recommendation=current_info.get("recommendationKey", "N/A").replace("_", " ").title(),
            target_price=round(current_info.get("targetMeanPrice", 0) or 0, 2),
            safety_score=safety_score,
            safety_label=safety_label,
            safety_color=safety_color,
            signals=signals,
            pnl_summary=pnl_summary,
            pred_rmse=round(pred_result.get("rmse", 0), 2),
            pred_mape=round(pred_result.get("mape", 0), 2),
            model_used=pred_result.get("model_used", "AI Prediction"),
            price_chart=price_chart,
            pnl_chart=pnl_chart,
            platforms=platforms,
            data_points=len(df_raw),
        )

    except Exception as e:
        traceback.print_exc()
        return render_template("index.html",
            error=f"An error occurred while analyzing '{query}': {str(e)}")


@app.route("/api/autocomplete")
def autocomplete():
    """Simple ticker suggestions."""
    q = request.args.get("q", "").upper()
    popular = [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "GOOGL", "name": "Alphabet (Google)"},
        {"ticker": "MSFT", "name": "Microsoft"},
        {"ticker": "AMZN", "name": "Amazon"},
        {"ticker": "TSLA", "name": "Tesla"},
        {"ticker": "NVDA", "name": "NVIDIA"},
        {"ticker": "META", "name": "Meta Platforms"},
        {"ticker": "NFLX", "name": "Netflix"},
        {"ticker": "AMD", "name": "AMD"},
        {"ticker": "INTC", "name": "Intel"},
        {"ticker": "PYPL", "name": "PayPal"},
        {"ticker": "UBER", "name": "Uber"},
        {"ticker": "ABNB", "name": "Airbnb"},
        {"ticker": "SHOP", "name": "Shopify"},
        {"ticker": "RELIANCE.NS", "name": "Reliance Industries"},
        {"ticker": "TCS.NS", "name": "Tata Consultancy Services"},
        {"ticker": "INFY.NS", "name": "Infosys"},
        {"ticker": "HDFCBANK.NS", "name": "HDFC Bank"},
    ]
    if q:
        results = [s for s in popular if q in s["ticker"] or q in s["name"].upper()]
    else:
        results = popular[:8]
    return jsonify(results)


@app.route("/dashboard")
def dashboard():
    """Dashboard showing search history and recent predictions."""
    try:
        history = get_search_history(limit=20)
        predictions = get_recent_predictions(limit=20)
        stats = get_dashboard_stats()
    except Exception:
        history, predictions, stats = [], [], {"total_searches": 0, "total_predictions": 0, "top_stocks": []}
    return render_template("dashboard.html", history=history, predictions=predictions, stats=stats)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
