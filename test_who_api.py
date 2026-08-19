import requests
import pandas as pd

# WHO GHO OData API - no API key needed
# Indicator: NCDMORT3070 = probability (%) of dying between age 30-70
# from cardiovascular disease, cancer, diabetes, or chronic respiratory disease
# This is WHO's standard cross-country comparable NCD/CVD mortality indicator.

BASE_URL = "https://ghoapi.azureedge.net/api/NCDMORT3070"

# A handful of African countries to start with (ISO-3 codes)
AFRICAN_COUNTRIES = ["ZAF", "NGA", "KEN", "EGY", "ETH", "GHA"]

def fetch_cvd_mortality():
    print("Requesting data from WHO GHO API...")
    response = requests.get(BASE_URL)
    response.raise_for_status()  # will raise an error if the request failed
    data = response.json()

    # The actual records live under the "value" key
    records = data["value"]
    print(f"Total records returned (all countries, all years): {len(records)}")

    df = pd.DataFrame(records)
    print(f"\nColumns available: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    return df

def filter_african_countries(df):
    # SpatialDim holds the ISO-3 country code in this API
    filtered = df[df["SpatialDim"].isin(AFRICAN_COUNTRIES)]
    print(f"\nRecords for our African countries: {len(filtered)}")
    print(filtered[["SpatialDim", "TimeDim", "Dim1", "NumericValue"]].head(20))
    return filtered

if __name__ == "__main__":
    df = fetch_cvd_mortality()
    african_df = filter_african_countries(df)