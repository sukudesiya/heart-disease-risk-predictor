import requests
import pandas as pd

AFRICAN_COUNTRIES = ["ZAF", "NGA", "KEN", "EGY", "ETH", "GHA"]

def fetch_gender_split_mortality():
    print("Fetching WHO CVD/NCD mortality data (by gender)...")
    url = "https://ghoapi.azureedge.net/api/NCDMORT3070"
    response = requests.get(url)
    response.raise_for_status()
    records = response.json()["value"]

    df = pd.DataFrame(records)
    df = df[df["SpatialDim"].isin(AFRICAN_COUNTRIES)]

    # This time keep ONLY male and female (drop "both sexes" - we want the
    # split, not the combined figure, for this chart)
    df = df[df["Dim1"].isin(["SEX_MLE", "SEX_FMLE"])]

    # Keep only the most recent year per country PER gender
    df = df.sort_values("TimeDim", ascending=False)
    df = df.drop_duplicates(subset=["SpatialDim", "Dim1"], keep="first")

    df = df[["SpatialDim", "Dim1", "TimeDim", "NumericValue"]].rename(columns={
        "SpatialDim": "country_code",
        "Dim1": "gender",
        "TimeDim": "year",
        "NumericValue": "mortality_rate",
    })

    # Make the gender labels readable
    df["gender"] = df["gender"].map({"SEX_MLE": "Male", "SEX_FMLE": "Female"})

    df = df.sort_values(["country_code", "gender"]).reset_index(drop=True)
    print(f"\nRecords: {len(df)}")
    print(df)
    return df

if __name__ == "__main__":
    df = fetch_gender_split_mortality()
    df.to_csv("africa_gender_mortality.csv", index=False)
    print("\nSaved to africa_gender_mortality.csv")