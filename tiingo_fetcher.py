def get_tiingo_metrics(ticker: str, api_key: str) -> dict:
    # Fetch Tiingo metrics for a given ticker
    # Return fundamentals, sentiment, news, etc.
    headers = {"Content-Type": "application/json"}
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}?token={api_key}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}