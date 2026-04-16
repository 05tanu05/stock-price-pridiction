"""
MongoDB integration for StockAI — stores search history and prediction results.
Requires a running MongoDB instance (default: localhost:27017).
"""

from datetime import datetime
from pymongo import MongoClient, DESCENDING

# ── Connection ──────────────────────────────────────────
_client = None
_db = None


def get_db():
    """Return the MongoDB database handle, connecting on first call."""
    global _client, _db
    if _db is None:
        _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        _db = _client["stockai"]
        # Create indexes for fast queries
        _db["search_history"].create_index([("searched_at", DESCENDING)])
        _db["predictions"].create_index([("predicted_at", DESCENDING)])
        _db["predictions"].create_index("ticker")
    return _db


# ── Search History ──────────────────────────────────────

def save_search(ticker, company_name, query):
    """Save a search event."""
    db = get_db()
    db["search_history"].insert_one({
        "ticker": ticker,
        "company_name": company_name,
        "query": query,
        "searched_at": datetime.utcnow(),
    })


def get_search_history(limit=20):
    """Return the most recent searches."""
    db = get_db()
    results = list(
        db["search_history"]
        .find({}, {"_id": 0})
        .sort("searched_at", DESCENDING)
        .limit(limit)
    )
    return results


# ── Prediction Results ──────────────────────────────────

def save_prediction(ticker, company_name, current_price, currency,
                    safety_score, safety_label, pnl_summary, model_used,
                    pred_mape):
    """Save a prediction snapshot for later comparison."""
    db = get_db()
    db["predictions"].insert_one({
        "ticker": ticker,
        "company_name": company_name,
        "current_price": current_price,
        "currency": currency,
        "safety_score": safety_score,
        "safety_label": safety_label,
        "pnl_summary": pnl_summary,
        "model_used": model_used,
        "pred_mape": pred_mape,
        "predicted_at": datetime.utcnow(),
    })


def get_recent_predictions(limit=20):
    """Return the most recent prediction results."""
    db = get_db()
    results = list(
        db["predictions"]
        .find({}, {"_id": 0})
        .sort("predicted_at", DESCENDING)
        .limit(limit)
    )
    return results


def get_dashboard_stats():
    """Return aggregate stats for the dashboard."""
    db = get_db()
    total_searches = db["search_history"].count_documents({})
    total_predictions = db["predictions"].count_documents({})

    # Most searched stocks (top 5)
    pipeline = [
        {"$group": {"_id": "$ticker", "company": {"$first": "$company_name"}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5},
    ]
    top_stocks = list(db["search_history"].aggregate(pipeline))

    return {
        "total_searches": total_searches,
        "total_predictions": total_predictions,
        "top_stocks": top_stocks,
    }
