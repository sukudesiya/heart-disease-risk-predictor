import requests
import pandas as pd
import time

AFRICAN_COUNTRIES = ["ZAF", "NGA", "KEN", "EGY", "ETH", "GHA"]

# ---------- WHO data ----------

def fetch_who_mortality():
    print("Fetching WHO CVD/NCD mortality data...")
    url = "https://ghoapi.azureedge.net/api/NCDMORT3070"
    response = requests.get(url)
    response.raise_for_status()
    records = response.json()["value"]

    df = pd.DataFrame(records)
    df = df[df["SpatialDim"].isin(AFRICAN_COUNTRIES)]

    # Keep only "both sexes" so we get one overall rate per country per year,
    # not separate male/female rows (we'll use the gender split later for
    # the demographic chart, but for the main KPI we want one number).
    df = df[df["Dim1"] == "SEX_BTSX"]

    # Keep only the most recent year available per country
    df = df.sort_values("TimeDim", ascending=False)
    df = df.drop_duplicates(subset="SpatialDim", keep="first")

    df = df[["SpatialDim", "TimeDim", "NumericValue"]].rename(columns={
        "SpatialDim": "country_code",
        "TimeDim": "mortality_year",
        "NumericValue": "cvd_mortality_rate",
    })
    print(f"WHO data ready: {len(df)} countries\n")
    return df

# ---------- World Bank data ----------

def fetch_worldbank_indicator(indicator_code, indicator_label, max_attempts=3):
    countries_str = ";".join(AFRICAN_COUNTRIES)
    url = f"https://api.worldbank.org/v2/country/{countries_str}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 500, "mrnev": 1}

    print(f"Fetching World Bank {indicator_label}...")
    for attempt in range(1, max_attempts + 1):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            break
        time.sleep(1)
    else:
        response.raise_for_status()

    records = response.json()[1]
    df = pd.DataFrame([{
        "country_code": r["countryiso3code"],
        "country_name": r["country"]["value"],
        indicator_label: r["value"],
    } for r in records])
    return df

# ---------- Join everything ----------

def build_combined_dataset():
    who_df = fetch_who_mortality()
    gdp_df = fetch_worldbank_indicator("NY.GDP.MKTP.CD", "gdp_usd")
    gdp_per_capita_df = fetch_worldbank_indicator("NY.GDP.PCAP.CD", "gdp_per_capita_usd")
    population_df = fetch_worldbank_indicator("SP.POP.TOTL", "population")

    # Start with GDP (has country_name) and merge everything else onto it,
    # joining on country_code (ISO-3) each time. Drop the duplicate
    # country_name column from gdp_per_capita_df before merging so we don't
    # end up with country_name_x / country_name_y.
    combined = gdp_df.merge(
        gdp_per_capita_df.drop(columns="country_name"), on="country_code", how="left"
    )
    combined = combined.merge(population_df[["country_code", "population"]], on="country_code", how="left")
    combined = combined.merge(who_df, on="country_code", how="left")

    print("\n=== Combined dataset ===")
    print(combined)

    print(f"\nAny missing values?\n{combined.isnull().sum()}")

    return combined

if __name__ == "__main__":
    df = build_combined_dataset()
    df.to_csv("africa_heart_data.csv", index=False)
    print("\nSaved to africa_heart_data.csv")