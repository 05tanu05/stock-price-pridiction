import time
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Common name-to-ticker mappings (comprehensive)
NAME_MAP = {
    # US Tech
    "APPLE": "AAPL", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "GOOGL": "GOOGL",
    "MICROSOFT": "MSFT", "AMAZON": "AMZN", "TESLA": "TSLA",
    "META": "META", "FACEBOOK": "META", "NETFLIX": "NFLX",
    "NVIDIA": "NVDA", "AMD": "AMD", "INTEL": "INTC",
    "TWITTER": "X", "UBER": "UBER", "AIRBNB": "ABNB",
    "SHOPIFY": "SHOP", "ZOOM": "ZM", "PAYPAL": "PYPL",
    "SPOTIFY": "SPOT", "SNAP": "SNAP", "SNAPCHAT": "SNAP",
    "PINTEREST": "PINS", "REDDIT": "RDDT", "ROBLOX": "RBLX",
    "PALANTIR": "PLTR", "SNOWFLAKE": "SNOW", "CROWDSTRIKE": "CRWD",
    "DATADOG": "DDOG", "CLOUDFLARE": "NET", "UNITY": "U",
    "ORACLE": "ORCL", "SALESFORCE": "CRM", "ADOBE": "ADBE",
    "IBM": "IBM", "CISCO": "CSCO", "QUALCOMM": "QCOM",
    "BROADCOM": "AVGO", "TEXAS INSTRUMENTS": "TXN",
    # US Finance / Consumer / Industrial
    "BERKSHIRE": "BRK-B", "JPMORGAN": "JPM", "JP MORGAN": "JPM",
    "GOLDMAN": "GS", "GOLDMAN SACHS": "GS", "MORGAN STANLEY": "MS",
    "BANK OF AMERICA": "BAC", "WELLS FARGO": "WFC", "CITIGROUP": "C",
    "VISA": "V", "MASTERCARD": "MA", "AMERICAN EXPRESS": "AXP",
    "COCA COLA": "KO", "COCA-COLA": "KO", "PEPSI": "PEP", "PEPSICO": "PEP",
    "DISNEY": "DIS", "WALT DISNEY": "DIS",
    "NIKE": "NKE", "STARBUCKS": "SBUX", "MCDONALD": "MCD", "MCDONALDS": "MCD",
    "WALMART": "WMT", "COSTCO": "COST", "TARGET": "TGT", "HOME DEPOT": "HD",
    "BOEING": "BA", "FORD": "F", "GM": "GM", "GENERAL MOTORS": "GM",
    "GENERAL ELECTRIC": "GE", "3M": "MMM", "CATERPILLAR": "CAT",
    # US Healthcare / Pharma
    "JOHNSON": "JNJ", "JOHNSON & JOHNSON": "JNJ", "PFIZER": "PFE",
    "MODERNA": "MRNA", "ABBVIE": "ABBV", "MERCK": "MRK",
    "ELI LILLY": "LLY", "LILLY": "LLY", "UNITEDHEALTH": "UNH",
    # US Energy
    "EXXON": "XOM", "EXXONMOBIL": "XOM", "CHEVRON": "CVX",
    # Indian Stocks
    "RELIANCE": "RELIANCE.NS", "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS", "TATA CONSULTANCY": "TCS.NS",
    "INFOSYS": "INFY.NS", "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HDFC": "HDFCBANK.NS", "HDFC BANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS", "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS", "STATE BANK": "SBIN.NS",
    "KOTAK": "KOTAKBANK.NS", "KOTAK BANK": "KOTAKBANK.NS",
    "BAJAJ": "BAJFINANCE.NS", "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "TATA": "TATAMOTORS.NS", "TATA MOTORS": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS", "TATA POWER": "TATAPOWER.NS",
    "ADANI": "ADANIENT.NS", "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS", "ADANI GREEN": "ADANIGREEN.NS",
    "AIRTEL": "BHARTIARTL.NS", "BHARTI AIRTEL": "BHARTIARTL.NS",
    "ITC": "ITC.NS", "HINDUSTAN UNILEVER": "HINDUNILVR.NS", "HUL": "HINDUNILVR.NS",
    "LARSEN": "LT.NS", "L&T": "LT.NS", "LARSEN & TOUBRO": "LT.NS",
    "MARUTI": "MARUTI.NS", "MARUTI SUZUKI": "MARUTI.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS", "MAHINDRA": "M&M.NS",
    "SUN PHARMA": "SUNPHARMA.NS", "DR REDDY": "DRREDDY.NS",
    "POWER GRID": "POWERGRID.NS", "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS", "COAL INDIA": "COALINDIA.NS",
    "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS",
    "NYKAA": "NYKAA.NS", "DMART": "DMART.NS",
    # Global
    "SAMSUNG": "005930.KS", "SONY": "SONY", "TOYOTA": "TM",
    "ALIBABA": "BABA", "BABA": "BABA", "TENCENT": "TCEHY",
    "TSM": "TSM", "TSMC": "TSM", "TAIWAN SEMICONDUCTOR": "TSM",
    "BITCOIN": "BTC-USD", "BTC": "BTC-USD",
    "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
    "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE OIL": "CL=F", "OIL": "CL=F",
}


def resolve_ticker(query):
    """Resolve a user query to a ticker symbol. Handles company names, partial matches."""
    query = query.strip()
    upper = query.upper()

    # Direct match in map
    if upper in NAME_MAP:
        return NAME_MAP[upper]

    # Partial match — check if the query is a substring of any key
    for name, ticker in NAME_MAP.items():
        if upper in name or name in upper:
            return ticker

    # If the query looks like a ticker already (short, alphanumeric + dots/dashes), return as-is
    return query


def _try_fetch(ticker_symbol, period):
    """Try fetching data for a single ticker symbol. Returns (ticker_symbol, company_name, df, info) or None."""
    try:
        ticker = yf.Ticker(ticker_symbol)

        with ThreadPoolExecutor(max_workers=2) as executor:
            history_future = executor.submit(ticker.history, period=period)
            info_future = executor.submit(lambda: ticker.info or {})

            df = history_future.result()

            try:
                info = info_future.result()
            except Exception:
                info = {}

        if df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        if df.empty:
            return None

        company_name = info.get("longName") or info.get("shortName") or ticker_symbol
        return ticker_symbol, company_name, df, info

    except Exception:
        return None


def fetch_all_data(query, period="1y", _retries=2):
    """
    Fetch price history and info in parallel for speed.
    Returns (ticker_symbol, company_name, df, info) or (None, None, None, None).
    Tries the resolved ticker first, then common suffixes (.NS, .BO) for Indian stocks.
    """
    ticker_symbol = resolve_ticker(query)

    for attempt in range(_retries):
        # Try the resolved ticker
        result = _try_fetch(ticker_symbol, period)
        if result:
            return result

        # If it failed and has no exchange suffix, try Indian exchanges
        if "." not in ticker_symbol:
            for suffix in [".NS", ".BO"]:
                result = _try_fetch(ticker_symbol + suffix, period)
                if result:
                    return result

        if attempt < _retries - 1:
            time.sleep(1)

    return None, None, None, None


def get_current_info(info):
    """Extract key current stock info."""
    keys = [
        "currentPrice", "regularMarketPrice", "previousClose",
        "open", "dayHigh", "dayLow", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "marketCap", "volume", "averageVolume", "trailingPE", "forwardPE",
        "dividendYield", "beta", "longName", "sector", "industry",
        "currency", "exchange", "shortName", "symbol",
        "recommendationKey", "targetMeanPrice", "returnOnEquity",
        "debtToEquity", "revenueGrowth", "earningsGrowth",
    ]
    result = {}
    for k in keys:
        val = info.get(k)
        if val is not None:
            result[k] = val
    return result
