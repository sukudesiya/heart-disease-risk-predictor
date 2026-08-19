import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="centered"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

# Train all models
@st.cache_resource
def train_models(_df):
    X = _df.drop("target", axis=1)
    y = _df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": m, "accuracy": acc}

    return results, X.columns.tolist()

@st.cache_data
def load_africa_data():
    return pd.read_csv("africa_heart_data.csv")

@st.cache_data
def load_gender_data():
    return pd.read_csv("africa_gender_mortality.csv")

@st.cache_data
def load_risk_factors():
    return pd.read_csv("africa_risk_factors.csv")

# Currency conversion setup - only used for the GDP KPI card
CURRENCY_MAP = {
    "ZAF": {"code": "ZAR", "symbol": "R"},
    "NGA": {"code": "NGN", "symbol": "₦"},
    "KEN": {"code": "KES", "symbol": "KSh"},
    "EGY": {"code": "EGP", "symbol": "E£"},
    "ETH": {"code": "ETB", "symbol": "Br"},
    "GHA": {"code": "GHS", "symbol": "GH₵"},
}

@st.cache_data(ttl=3600)  # refresh rates once an hour at most
def load_exchange_rates():
    import requests
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        response.raise_for_status()
        return response.json()["rates"]
    except Exception:
        # If the currency API is unreachable, fall back to USD everywhere
        # rather than crashing the app.
        return None

# Load and train (shared across both tabs)
df = load_data()
all_results, feature_names = train_models(df)
africa_df = load_africa_data()
gender_df = load_gender_data()
risk_df = load_risk_factors()

# Find best model
best_name = max(all_results, key=lambda k: all_results[k]["accuracy"])

# Sidebar - model comparison
st.sidebar.header("Model Comparison")
comparison_data = {
    "Model": list(all_results.keys()),
    "Accuracy": [f"{all_results[k]['accuracy']:.1%}" for k in all_results],
}
comparison_df = pd.DataFrame(comparison_data)
st.sidebar.dataframe(comparison_df, hide_index=True)
st.sidebar.write(f"Best performer: {best_name}")

# Model selector
selected_model_name = st.sidebar.selectbox(
    "Choose prediction model",
    options=list(all_results.keys()),
    index=list(all_results.keys()).index(best_name)
)

model = all_results[selected_model_name]["model"]
accuracy = all_results[selected_model_name]["accuracy"]

st.sidebar.metric("Selected Model Accuracy", f"{accuracy:.1%}")
st.sidebar.write(f"Trained on {len(df)} patient records")

# Pin the branding caption to the very bottom of the sidebar
st.markdown(
    """
    <style>
    .sidebar-footer {
        position: fixed;
        bottom: 1rem;
        left: 0;
        width: 21rem;
        padding-left: 1.5rem;
        color: rgba(49, 51, 63, 0.6);
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div class='sidebar-footer'>Built by Siyanda Tshakaza</div>",
    unsafe_allow_html=True
)

# Main title
st.title("Heart Disease Risk Predictor")
st.caption("A machine learning tool for individual risk prediction, combined with real-world African CVD context.")

# ============================================
# TABS
# ============================================
tab1, tab2 = st.tabs(["🫀 Risk Predictor", "🌍 Africa Context"])

# ============================================
# TAB 1: PATIENT RISK PREDICTOR
# ============================================
with tab1:
    st.write("Enter patient information below to predict heart disease risk.")

    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x]
        )

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            options=[0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
        )
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x]
        )

    # Prediction
    if st.button("Predict Risk", type="primary"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.divider()
        st.header("Prediction Results")
        st.caption(f"Using: {selected_model_name}")

        # Convert probability to a percentage
        risk_percentage = probability[1] * 100

        # Show color-coded message based on prediction
        if prediction == 1:
            st.error(f"High Risk of Heart Disease ({risk_percentage:.1f}% probability)")
        else:
            st.success(f"Low Risk of Heart Disease ({risk_percentage:.1f}% probability)")

        # Risk level indicator
        st.subheader("Risk Level")
        st.progress(risk_percentage / 100)

        if risk_percentage < 30:
            st.write("Risk Level: LOW")
        elif risk_percentage < 60:
            st.write("Risk Level: MODERATE")
        else:
            st.write("Risk Level: HIGH")

        # Feature importance (only for tree-based models)
        if hasattr(model, "feature_importances_"):
            st.subheader("Key Risk Factors")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values("Importance", ascending=False).head(5)

            st.bar_chart(importance_df.set_index("Feature"))
        else:
            st.subheader("Model Coefficients")
            coefficients = model.coef_[0]
            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Weight": np.abs(coefficients)
            }).sort_values("Weight", ascending=False).head(5)

            st.bar_chart(coef_df.set_index("Feature"))

        st.caption("This tool is for educational purposes only. Always consult a medical professional.")

# ============================================
# TAB 2: AFRICA HEART DISEASE CONTEXT
# ============================================
with tab2:
    st.write("Country-level heart disease context across Africa, combining WHO health data with World Bank economic data.")

    # Country filter
    selected_country = st.selectbox(
        "Select a country",
        options=africa_df["country_name"].tolist()
    )

    filtered_df = africa_df[africa_df["country_name"] == selected_country]
    country_data = filtered_df.iloc[0]  # grab the single matching row as a Series

    # Convert GDP to the selected country's local currency, where possible
    exchange_rates = load_exchange_rates()
    country_code = country_data["country_code"]

    if exchange_rates is not None and country_code in CURRENCY_MAP:
        currency_info = CURRENCY_MAP[country_code]
        rate = exchange_rates.get(currency_info["code"])
        if rate is not None:
            gdp_local = country_data["gdp_usd"] * rate
            gdp_display = f"{currency_info['symbol']}{gdp_local / 1_000_000_000:,.1f}B"
        else:
            gdp_display = f"${country_data['gdp_usd'] / 1_000_000_000:,.1f}B"
    else:
        gdp_display = f"${country_data['gdp_usd'] / 1_000_000_000:,.1f}B"

    # KPI cards
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("Population", f"{country_data['population']:,.0f}")

    with kpi2:
        st.metric("GDP", gdp_display)

    with kpi3:
        st.metric("CVD Mortality Rate", f"{country_data['cvd_mortality_rate']:.1f}%")

    st.caption(f"Mortality data year: {country_data['mortality_year']} | GDP per capita (USD): ${country_data['gdp_per_capita_usd']:,.0f}")
    st.caption("GDP shown in local currency where available. Live exchange rates via open.er-api.com.")

    st.divider()

    # ---- Grouped bar chart: mortality rate by gender across countries ----
    st.subheader("CVD Mortality Rate by Gender")

    # Pivot so countries are rows and Male/Female become separate columns -
    # st.bar_chart will automatically draw one bar per column, grouped per country
    gender_pivot = gender_df.pivot(index="country_code", columns="gender", values="mortality_rate")

    st.bar_chart(gender_pivot)
    st.caption("CVD/NCD premature mortality rate (%) by gender, most recent available year per country.")

    st.divider()

    # ---- Pie/donut chart: major risk factors for selected country ----
    st.subheader("Major Risk Factors")

    # Match the risk factor row to whichever country is selected above
    country_code_lookup = africa_df.set_index("country_name")["country_code"].to_dict()
    selected_code = country_code_lookup[selected_country]
    risk_row = risk_df[risk_df["country_code"] == selected_code].iloc[0]

    risk_chart_data = pd.DataFrame({
        "Risk Factor": ["Hypertension", "Obesity", "Smoking"],
        "Prevalence (%)": [risk_row["hypertension"], risk_row["obesity"], risk_row["smoking"]]
    })

    fig = px.pie(
        risk_chart_data,
        names="Risk Factor",
        values="Prevalence (%)",
        hole=0.4,
        title=f"Risk Factor Prevalence - {selected_country}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each slice reflects that risk factor's prevalence (%) among adults, not a share of 100%. "
               "Diet and Rheumatic Heart Disease are not shown here as WHO does not publish a single "
               "comparable indicator for these via this API.")

    st.divider()

    # ---- Scatter chart: GDP per capita vs CVD mortality rate, all countries ----
    st.subheader("GDP vs. CVD Mortality Rate Across Africa")

    fig_scatter = px.scatter(
        africa_df,
        x="gdp_per_capita_usd",
        y="cvd_mortality_rate",
        size="population",
        color="country_name",
        text="country_name",
        labels={
            "gdp_per_capita_usd": "GDP per Capita (USD)",
            "cvd_mortality_rate": "CVD Mortality Rate (%)",
            "country_name": "Country"
        },
        title="GDP per Capita vs. CVD Mortality Rate"
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Bubble size represents population. This compares all 6 countries regardless of the filter selected above.")

    st.divider()

    # ---- Insight callout: which countries carry the highest burden ----
    st.subheader("Key Insight: Highest Burden Countries")

    burden_ranked = africa_df.sort_values("cvd_mortality_rate", ascending=False).reset_index(drop=True)
    highest = burden_ranked.iloc[0]
    lowest = burden_ranked.iloc[-1]

    st.info(
        f"**{highest['country_name']}** has the highest CVD mortality rate among these {len(burden_ranked)} "
        f"African countries at **{highest['cvd_mortality_rate']:.1f}%**, "
        f"followed by **{burden_ranked.iloc[1]['country_name']}** at **{burden_ranked.iloc[1]['cvd_mortality_rate']:.1f}%**. "
        f"**{lowest['country_name']}** has the lowest rate at **{lowest['cvd_mortality_rate']:.1f}%**. "
        f"Notably, GDP per capita alone does not explain this ranking — "
        f"{highest['country_name']}'s GDP per capita (\\${highest['gdp_per_capita_usd']:,.0f}) is lower than "
        f"South Africa's (\\${africa_df[africa_df['country_name'] == 'South Africa']['gdp_per_capita_usd'].values[0]:,.0f}), "
        f"yet its mortality rate is higher, suggesting lifestyle and risk-factor prevalence "
        f"(e.g. hypertension, obesity) may play a larger role than national wealth alone."
    )

    with st.expander("See full country comparison table"):
        st.dataframe(
            burden_ranked[["country_name", "cvd_mortality_rate", "gdp_per_capita_usd", "population"]]
            .rename(columns={
                "country_name": "Country",
                "cvd_mortality_rate": "CVD Mortality Rate (%)",
                "gdp_per_capita_usd": "GDP per Capita (USD)",
                "population": "Population"
            }),
            hide_index=True
        )

    st.divider()

    st.markdown(
        """
        <style>
        [data-testid="stDownloadButton"] button {
            background-color: #90CAF9;
            color: #0d3b66;
            border: none;
        }
        [data-testid="stDownloadButton"] button:hover {
            background-color: #6FB8F5;
            color: #0d3b66;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        label="Download Africa CVD Data (CSV)",
        data=africa_df.to_csv(index=False),
        file_name="africa_heart_disease_data.csv",
        mime="text/csv"
    )