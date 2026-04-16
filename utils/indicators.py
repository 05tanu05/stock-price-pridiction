import pandas as pd
import numpy as np


def analyze_safety(df, info):
    """
    Simplified safety analysis based on price trends and fundamentals.
    Returns a safety score (0-100), label, and detailed signals.
    """
    signals = []
    score = 50  # neutral start

    closes = df["Close"].values

    # --- Price Trend (30-day) ---
    if len(closes) >= 30:
        price_30d_ago = closes[-30]
        price_now = closes[-1]
        change_30d = ((price_now - price_30d_ago) / price_30d_ago) * 100
        if change_30d > 5:
            signals.append({"indicator": "30-Day Trend", "value": f"{change_30d:+.1f}%", "signal": "Price is rising", "status": "success"})
            score += 10
        elif change_30d < -5:
            signals.append({"indicator": "30-Day Trend", "value": f"{change_30d:+.1f}%", "signal": "Price is falling", "status": "danger"})
            score -= 10
        else:
            signals.append({"indicator": "30-Day Trend", "value": f"{change_30d:+.1f}%", "signal": "Price is stable", "status": "warning"})
            score += 3

    # --- 52-Week Position ---
    week52_high = info.get("fiftyTwoWeekHigh", df["High"].max())
    week52_low = info.get("fiftyTwoWeekLow", df["Low"].min())
    current = closes[-1]
    if week52_high and week52_low and (week52_high - week52_low) > 0:
        position = ((current - week52_low) / (week52_high - week52_low)) * 100
        if position > 80:
            signals.append({"indicator": "52-Week Position", "value": f"Near High", "signal": "Trading near 52-week high — may be expensive", "status": "warning"})
            score -= 5
        elif position < 20:
            signals.append({"indicator": "52-Week Position", "value": f"Near Low", "signal": "Trading near 52-week low — could be a bargain or risky", "status": "warning"})
            score += 2
        else:
            signals.append({"indicator": "52-Week Position", "value": f"Mid Range", "signal": "Trading in a healthy range", "status": "success"})
            score += 5

    # --- Volatility (simple: std of last 30 days) ---
    if len(closes) >= 30:
        daily_returns = np.diff(closes[-30:]) / closes[-30:-1]
        volatility = np.std(daily_returns) * 100
        if volatility < 1.5:
            signals.append({"indicator": "Volatility", "value": f"{volatility:.1f}%", "signal": "Low volatility — Stable stock", "status": "success"})
            score += 8
        elif volatility > 3.0:
            signals.append({"indicator": "Volatility", "value": f"{volatility:.1f}%", "signal": "High volatility — Risky", "status": "danger"})
            score -= 8
        else:
            signals.append({"indicator": "Volatility", "value": f"{volatility:.1f}%", "signal": "Moderate volatility", "status": "warning"})
            score += 3

    # --- P/E Ratio ---
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe:
        if pe < 15:
            signals.append({"indicator": "P/E Ratio", "value": f"{pe:.1f}", "signal": "Undervalued — Good price", "status": "success"})
            score += 8
        elif pe > 40:
            signals.append({"indicator": "P/E Ratio", "value": f"{pe:.1f}", "signal": "Overvalued — Expensive", "status": "danger"})
            score -= 8
        else:
            signals.append({"indicator": "P/E Ratio", "value": f"{pe:.1f}", "signal": "Fair valued", "status": "success"})
            score += 4

    # --- Beta ---
    beta = info.get("beta")
    if beta is not None:
        if beta < 0.8:
            signals.append({"indicator": "Risk Level", "value": f"Low (Beta {beta:.2f})", "signal": "Less risky than market", "status": "success"})
            score += 8
        elif beta > 1.5:
            signals.append({"indicator": "Risk Level", "value": f"High (Beta {beta:.2f})", "signal": "More risky than market", "status": "danger"})
            score -= 8
        else:
            signals.append({"indicator": "Risk Level", "value": f"Medium (Beta {beta:.2f})", "signal": "Similar to market risk", "status": "warning"})
            score += 3

    # --- Analyst Recommendation ---
    rec = info.get("recommendationKey", "").lower()
    rec_map = {
        "strong_buy": ("Strong Buy", "success", 15),
        "buy": ("Buy", "success", 10),
        "hold": ("Hold", "warning", 0),
        "sell": ("Sell", "danger", -10),
        "strong_sell": ("Strong Sell", "danger", -15),
    }
    if rec in rec_map:
        label, status, adj = rec_map[rec]
        signals.append({"indicator": "Analyst Opinion", "value": label, "signal": f"Wall Street analysts say: {label}", "status": status})
        score += adj

    # Clamp score
    score = max(0, min(100, score))

    if score >= 65:
        safety_label = "SAFE"
        safety_color = "success"
    elif score >= 45:
        safety_label = "MODERATE RISK"
        safety_color = "warning"
    else:
        safety_label = "HIGH RISK"
        safety_color = "danger"

    return score, safety_label, safety_color, signals


def get_buying_platforms(ticker_symbol, current_price=0, safety_score=50, volatility_pct=2.0, pe_ratio=0, recommendation=""):
    """
    Return top 5 platforms ranked by user benefit for this specific stock.
    Considers: market (Indian/US), price level, risk, volatility, and valuation.
    """
    is_indian = ticker_symbol.endswith(".NS") or ticker_symbol.endswith(".BO")
    is_safe = safety_score >= 65
    is_risky = safety_score < 45
    is_volatile = volatility_pct > 3.0
    is_cheap = current_price < 500 if not is_indian else current_price < 1000
    is_undervalued = 0 < pe_ratio < 15

    if is_indian:
        all_platforms = [
            {
                "name": "Zerodha",
                "desc": "India's #1 discount broker — lowest brokerage",
                "url": "https://zerodha.com",
                "icon": "Z",
                "rank_score": 90,
                "benefits": [
                    "Zero brokerage on equity delivery trades",
                    "Advanced Kite charting for technical analysis",
                    "Varsity — free learning modules for beginners",
                ],
            },
            {
                "name": "Groww",
                "desc": "Simplest app for beginners to start investing",
                "url": "https://groww.in",
                "icon": "G",
                "rank_score": 80,
                "benefits": [
                    "Zero account opening & maintenance charges",
                    "Stocks + mutual funds + FDs in one app",
                    "Instant KYC — start trading in 10 minutes",
                ],
            },
            {
                "name": "Upstox",
                "desc": "Fast execution with ultra-low fees",
                "url": "https://upstox.com",
                "icon": "U",
                "rank_score": 75,
                "benefits": [
                    "Rs.20 flat per trade for intraday & F&O",
                    "Pro charting tools with TradingView integration",
                    "Free demat account with no hidden charges",
                ],
            },
            {
                "name": "Angel One",
                "desc": "Full-service broker with expert research",
                "url": "https://www.angelone.in",
                "icon": "A",
                "rank_score": 70,
                "benefits": [
                    "Free research reports & stock recommendations",
                    "SmartAPI for algo trading enthusiasts",
                    "Margin trading facility available",
                ],
            },
            {
                "name": "Paytm Money",
                "desc": "Invest directly from your Paytm app",
                "url": "https://www.paytmmoney.com",
                "icon": "P",
                "rank_score": 65,
                "benefits": [
                    "Seamless UPI payments — instant fund transfer",
                    "SIP in stocks starting from small amounts",
                    "Simple UI — perfect for first-time investors",
                ],
            },
        ]

        # Adjust scores based on stock characteristics
        for p in all_platforms:
            if is_safe and p["name"] == "Groww":
                p["rank_score"] += 10  # beginners benefit most from safe stocks
                p["benefits"].insert(0, f"This stock has a safety score of {safety_score} — great for beginners on Groww")
            if is_risky and p["name"] == "Zerodha":
                p["rank_score"] += 10  # advanced tools needed for risky stocks
                p["benefits"].insert(0, "Advanced stop-loss & GTT orders to manage this stock's risk")
            if is_volatile and p["name"] == "Upstox":
                p["rank_score"] += 10  # fast execution matters for volatile stocks
                p["benefits"].insert(0, "Ultra-fast execution crucial for this volatile stock")
            if is_undervalued and p["name"] == "Angel One":
                p["rank_score"] += 10
                p["benefits"].insert(0, "Angel One research confirms undervalued stocks like this")
            if is_cheap and p["name"] == "Paytm Money":
                p["rank_score"] += 8
                p["benefits"].insert(0, f"Affordable at {chr(8377)}{current_price:.0f} — easy to start with small investments")

    else:
        all_platforms = [
            {
                "name": "Fidelity",
                "desc": "Best for long-term investors — zero commissions",
                "url": "https://www.fidelity.com",
                "icon": "F",
                "rank_score": 90,
                "benefits": [
                    "Zero commission on US stocks & ETFs",
                    "Fractional shares — invest with as little as $1",
                    "Top-rated research & retirement planning tools",
                ],
            },
            {
                "name": "Charles Schwab",
                "desc": "All-in-one investing + banking platform",
                "url": "https://www.schwab.com",
                "icon": "S",
                "rank_score": 85,
                "benefits": [
                    "Zero commission + no account minimums",
                    "Thinkorswim platform for advanced charting",
                    "24/7 trading on select stocks & ETFs",
                ],
            },
            {
                "name": "Robinhood",
                "desc": "Commission-free trading with simple UI",
                "url": "https://robinhood.com",
                "icon": "R",
                "rank_score": 80,
                "benefits": [
                    "Zero commission — keep 100% of your profits",
                    "Fractional shares from $1",
                    "Instant deposits — start trading immediately",
                ],
            },
            {
                "name": "Interactive Brokers",
                "desc": "Best for global market access & low margin rates",
                "url": "https://www.interactivebrokers.com",
                "icon": "I",
                "rank_score": 75,
                "benefits": [
                    "Access to 150+ global markets from one account",
                    "Lowest margin rates in the industry (from 1.5%)",
                    "Professional-grade trading tools & analytics",
                ],
            },
            {
                "name": "Webull",
                "desc": "Advanced tools with zero commissions",
                "url": "https://www.webull.com",
                "icon": "W",
                "rank_score": 70,
                "benefits": [
                    "Zero commission + extended hours trading",
                    "Advanced charting with 50+ technical indicators",
                    "Paper trading to practice risk-free",
                ],
            },
        ]

        # Adjust scores based on stock characteristics
        for p in all_platforms:
            if is_safe and p["name"] == "Fidelity":
                p["rank_score"] += 10
                p["benefits"].insert(0, f"Safety score {safety_score} — ideal for Fidelity's buy-and-hold approach")
            if is_safe and p["name"] == "Robinhood":
                p["rank_score"] += 8
                p["benefits"].insert(0, "Safe stock + zero commission = maximum returns for beginners")
            if is_risky and p["name"] == "Interactive Brokers":
                p["rank_score"] += 12
                p["benefits"].insert(0, "Advanced risk management tools essential for this high-risk stock")
            if is_risky and p["name"] == "Charles Schwab":
                p["rank_score"] += 8
                p["benefits"].insert(0, "Schwab's research helps navigate this stock's high risk level")
            if is_volatile and p["name"] == "Webull":
                p["rank_score"] += 10
                p["benefits"].insert(0, "Extended hours trading helps manage this stock's high volatility")
            if is_volatile and p["name"] == "Interactive Brokers":
                p["rank_score"] += 8
                p["benefits"].insert(0, "Advanced order types (stop-limit, trailing stop) for volatile stocks")
            if is_undervalued and p["name"] == "Fidelity":
                p["rank_score"] += 8
                p["benefits"].insert(0, "Undervalued stock (P/E {:.1f}) — Fidelity research tools can confirm".format(pe_ratio))
            if is_cheap and p["name"] == "Robinhood":
                p["rank_score"] += 8
                p["benefits"].insert(0, f"At ${current_price:.2f}, fractional shares let you start with just $1")

    # Sort by rank_score descending, return top 5
    all_platforms.sort(key=lambda x: x["rank_score"], reverse=True)
    # Add rank position and trim benefits to top 3
    for i, p in enumerate(all_platforms):
        p["rank"] = i + 1
        p["benefits"] = p["benefits"][:3]
    return all_platforms[:5]
