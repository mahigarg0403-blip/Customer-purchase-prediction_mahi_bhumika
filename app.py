import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Purchase Predictor",
    page_icon="🛒",
    layout="centered"
)

# ─── Load model and data ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/random_forest.pkl')

@st.cache_data
def load_background():
    df = pd.read_csv('data/x_test_tree_final.csv')
    return df.sample(100, random_state=42)

rf_model = load_model()
background_data = load_background()

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Customer Purchase Predictor")
st.markdown("**Predict whether an e-commerce session will result in a purchase**")
st.markdown("*Using Random Forest — best performing model (AUC: 0.9255)*")
st.divider()

# ─── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Session Details")
st.markdown("Enter the session information below:")

col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input(
        "Administrative Pages Visited",
        min_value=0, max_value=50, value=2,
        help="Number of administrative pages visited")

    administrative_duration = st.number_input(
        "Administrative Duration (seconds)",
        min_value=0.0, max_value=5000.0, value=50.0,
        help="Total time spent on administrative pages")

    informational = st.number_input(
        "Informational Pages Visited",
        min_value=0, max_value=50, value=1,
        help="Number of informational pages visited")

    informational_duration = st.number_input(
        "Informational Duration (seconds)",
        min_value=0.0, max_value=5000.0, value=20.0,
        help="Total time spent on informational pages")

    product_related = st.number_input(
        "Product Pages Visited",
        min_value=0, max_value=700, value=10,
        help="Number of product related pages visited")

    product_related_duration = st.number_input(
        "Product Duration (seconds)",
        min_value=0.0, max_value=50000.0, value=500.0,
        help="Total time spent on product pages")


with col2:
    exit_rates = st.slider(
        "Exit Rate",
        min_value=0.0, max_value=0.2, value=0.04,
        help="Average exit rate of visited pages")

    page_values = st.number_input(
        "Page Value",
        min_value=0.0, max_value=400.0, value=15.0,
        help="Average page value of visited pages")

    special_day = st.slider(
        "Special Day Proximity",
        min_value=0.0, max_value=1.0, value=0.0,
        help="Closeness to a special day (0=not close, 1=very close)")

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                'Jul','Aug','Sep','Oct','Nov','Dec'][x-1],
        index=10,
        help="Month of the session")

    operating_systems = st.selectbox(
        "Operating System",
        options=[1, 2, 3, 4, 5, 6, 7, 8],
        index=1)

    browser = st.selectbox(
        "Browser",
        options=list(range(1, 14)),
        index=1)

    region = st.selectbox(
        "Region",
        options=list(range(1, 10)),
        index=0)

    traffic_type = st.selectbox(
        "Traffic Type",
        options=list(range(1, 21)),
        index=1)

    visitor_type = st.selectbox(
        "Visitor Type",
        options=[0, 1, 2],
        format_func=lambda x: ['New Visitor', 'Other', 'Returning Visitor'][x],
        index=2)

    weekend = st.radio(
        "Weekend Session?",
        options=[0, 1],
        format_func=lambda x: 'No' if x == 0 else 'Yes',
        horizontal=True)

# ─── Feature engineering (must match Step 4 exactly) ──────────────────────────
def engineer_features(inputs):
    d = inputs.copy()

    # Interaction_Value = ProductRelated x PageValues
    d['Interaction_Value'] = d['ProductRelated'] * d['PageValues']

    # PageValues_bin using actual training median of non-zero values
    PAGEVALUES_MEDIAN = 20.624306060000002
    if d['PageValues'] == 0:
        d['PageValues_bin'] = 0
    elif d['PageValues'] <= PAGEVALUES_MEDIAN:
        d['PageValues_bin'] = 1
    else:
        d['PageValues_bin'] = 2

    # ProductRelated_Duration_log
    d['ProductRelated_Duration_log'] = np.log1p(d['ProductRelated_Duration'])

    # Session_Intensity
    if d['ProductRelated'] != 0:
        d['Session_Intensity'] = d['ProductRelated_Duration'] / d['ProductRelated']
    else:
        d['Session_Intensity'] = 0

    # November_New = November AND New Visitor (VisitorType == 0)
    d['November_New'] = int(d['Month'] == 11 and d['VisitorType'] == 0)

    return d
# ─── Predict button ────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Predict Purchase Intent", use_container_width=True)

if predict_btn:
    # Build raw input dict
    raw_inputs = {
        'Administrative': administrative,
        'Administrative_Duration': administrative_duration,
        'Informational': informational,
        'Informational_Duration': informational_duration,
        'ProductRelated': product_related,
        'ProductRelated_Duration': product_related_duration,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': special_day,
        'Month': month,
        'OperatingSystems': operating_systems,
        'Browser': browser,
        'Region': region,
        'TrafficType': traffic_type,
        'VisitorType': visitor_type,
        'Weekend': weekend
    }

    # Engineer features
    engineered = engineer_features(raw_inputs)

    # Build dataframe in correct column order
    feature_order = background_data.columns.tolist()
    input_df = pd.DataFrame([engineered])[feature_order]

    # Predict
    prob = rf_model.predict_proba(input_df)[0][1]
    prediction = int(prob >= 0.5)

    # ─── Result display ────────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success(f"✅ **HIGH purchase intent detected**")
    else:
        st.error(f"❌ **LOW purchase intent detected**")

    st.metric(
        label="Purchase Probability",
        value=f"{prob*100:.1f}%",
        delta=f"{'Above' if prob >= 0.5 else 'Below'} 50% threshold"
    )

    # ─── SHAP explanation ──────────────────────────────────────────────────────
    st.subheader("🔍 Why this prediction?")
    st.markdown("*SHAP values show which features pushed the prediction up or down*")

    with st.spinner("Calculating SHAP explanation..."):
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[:, :, 1][0]

        # Build SHAP dataframe
        shap_df = pd.DataFrame({
            'Feature': feature_order,
            'Value': input_df.iloc[0].values,
            'SHAP': shap_vals
        }).sort_values('SHAP', key=abs, ascending=False).head(10)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_df['SHAP']]
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on prediction)')
        ax.set_title('Top 10 Features Driving This Prediction\n'
                     '(Green = pushes toward purchase, Red = pushes away)')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        # Feature value table
        st.markdown("**Feature values for this session:**")
        display_df = shap_df[['Feature', 'Value', 'SHAP']].copy()
        display_df['SHAP'] = display_df['SHAP'].round(4)
        display_df['Value'] = display_df['Value'].round(4)
        st.dataframe(display_df, use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Built by Mahi & Bhumika | "
    "UCI Online Shoppers Dataset | "
    "Random Forest (AUC: 0.9255)*"
)