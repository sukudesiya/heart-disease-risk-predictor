import requests

# open.er-api.com - free, keyless exchange rate API
# Docs: https://www.exchangerate-api.com/docs/free
# Note: this free tier asks for attribution (a link back to them), which
# we'll include as a small caption in the app.

# Map each of our countries to its local currency code + symbol
CURRENCY_MAP = {
    "ZAF": {"code": "ZAR", "symbol": "R"},
    "NGA": {"code": "NGN", "symbol": "₦"},
    "KEN": {"code": "KES", "symbol": "KSh"},
    "EGY": {"code": "EGP", "symbol": "E£"},
    "ETH": {"code": "ETB", "symbol": "Br"},
    "GHA": {"code": "GHS", "symbol": "GH₵"},
}

def fetch_exchange_rates():
    print("Fetching exchange rates (base: USD)...")
    url = "https://open.er-api.com/v6/latest/USD"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    print(f"Result: {data.get('result')}")
    print(f"Last update: {data.get('time_last_update_utc')}")

    rates = data["rates"]

    print("\nRates for our countries' currencies:")
    for country, info in CURRENCY_MAP.items():
        code = info["code"]
        if code in rates:
            print(f"  {country} ({code}): 1 USD = {rates[code]} {code}")
        else:
            print(f"  {country} ({code}): NOT FOUND in API response")

    return rates

if __name__ == "__main__":
    rates = fetch_exchange_rates()

    # Quick sanity check: convert South Africa's known GDP figure
    sample_gdp_usd = 427_184_320_000  # from our existing data
    zar_rate = rates["ZAR"]
    converted = sample_gdp_usd * zar_rate
    print(f"\nSanity check - South Africa GDP:")
    print(f"  ${sample_gdp_usd:,.0f} USD -> R{converted:,.0f} ZAR")