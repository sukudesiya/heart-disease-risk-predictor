import requests
import pandas as pd

AFRICAN_COUNTRIES = ["ZAF", "NGA", "KEN", "EGY", "ETH", "GHA"]

# Risk factor indicators (all real WHO GHO indicators, no API key needed)
RISK_FACTORS = {
    "NCD_HYP_PREVALENCE_A": "Hypertension",
    "NCD_BMI_30A": "Obesity",
    "M_Est_tob_curr_std": "Smoking",
}

def fetch_risk_factor(indicator_code, label):
    print(f"Fetching {label} ({indicator_code})...")
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"  -> FAILED (status {response.status_code}) - skipping this indicator")
        return None

    records = response.json()["value"]
    df = pd.DataFrame(records)

    if df.empty or "SpatialDim" not in df.columns:
        print(f"  -> No usable data returned - skipping this indicator")
        return None

    df = df[df["SpatialDim"].isin(AFRICAN_COUNTRIES)]

    # Try to keep only "both sexes" combined if a sex dimension exists
    if "Dim1" in df.columns:
        both_sexes_values = ["SEX_BTSX", "BTSX", "TOTAL"]
        if df["Dim1"].isin(both_sexes_values).any():
            df = df[df["Dim1"].isin(both_sexes_values)]

    if df.empty:
        print(f"  -> No African country data found for this indicator - skipping")
        return None

    # Keep most recent year per country
    df = df.sort_values("TimeDim", ascending=False)
    df = df.drop_duplicates(subset="SpatialDim", keep="first")

    df = df[["SpatialDim", "TimeDim", "NumericValue"]].rename(columns={
        "SpatialDim": "country_code",
        "TimeDim": "year",
        "NumericValue": label.lower(),
    })
    print(f"  -> Got {len(df)} country rows")
    return df

if __name__ == "__main__":
    all_data = {}
    for code, label in RISK_FACTORS.items():
        result = fetch_risk_factor(code, label)
        if result is not None:
            all_data[label] = result

    print(f"\nSuccessfully fetched: {list(all_data.keys())}")

    # Merge all risk factors into one combined table, joined on country_code
    combined = None
    for label, df in all_data.items():
        df_slim = df[["country_code", label.lower()]]
        combined = df_slim if combined is None else combined.merge(df_slim, on="country_code", how="outer")

    print("\n--- Combined risk factors ---")
    print(combined)

    combined.to_csv("africa_risk_factors.csv", index=False)
    print("\nSaved to africa_risk_factors.csv")