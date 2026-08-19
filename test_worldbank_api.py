import requests
import pandas as pd
import time

# World Bank Indicators API - no API key needed
# Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898599
#
# Indicator codes we need:
#   NY.GDP.MKTP.CD  = GDP (current US$)
#   NY.GDP.PCAP.CD  = GDP per capita (current US$)
#   SP.POP.TOTL     = Population, total

BASE_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"

# Same African countries as Step 1 (ISO-3 codes)
AFRICAN_COUNTRIES = ["ZAF", "NGA", "KEN", "EGY", "ETH", "GHA"]

def response_url_preview(url, params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{url}?{query}"

def fetch_indicator(indicator_code, indicator_label):
    countries_str = ";".join(AFRICAN_COUNTRIES)
    url = BASE_URL.format(countries=countries_str, indicator=indicator_code)

    params = {
        "format": "json",
        "per_page": 500,
        "mrnev": 1,  # most recent non-empty value only, per country
    }

    print(f"\nRequesting {indicator_label} ({indicator_code})...")
    print(f"Full URL: {response_url_preview(url, params)}")

    # World Bank API occasionally returns a transient server error (HTML page
    # instead of JSON). Retry a couple of times before giving up.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            break
        print(f"Attempt {attempt} failed (status {response.status_code}), retrying...")
        time.sleep(1)
    else:
        print(f"ERROR - Status code: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()

    data = response.json()

    # World Bank API returns a list: [metadata, records]
    metadata = data[0]
    records = data[1]

    print(f"Total records returned: {metadata['total']}")

    rows = []
    for r in records:
        rows.append({
            "country_code": r["countryiso3code"],
            "country_name": r["country"]["value"],
            "year": r["date"],
            indicator_label: r["value"],
        })

    df = pd.DataFrame(rows)
    print(df)
    return df

if __name__ == "__main__":
    gdp_df = fetch_indicator("NY.GDP.MKTP.CD", "gdp_usd")
    gdp_per_capita_df = fetch_indicator("NY.GDP.PCAP.CD", "gdp_per_capita_usd")
    population_df = fetch_indicator("SP.POP.TOTL", "population")