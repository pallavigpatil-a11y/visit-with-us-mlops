
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Scikit-learn compatibility fix
# ---------------------------------------------------------
# This handles models saved with a ColumnTransformer that
# contains the internal _RemainderColsList object.
# ---------------------------------------------------------

import sklearn.compose._column_transformer as _ct_mod

if not hasattr(_ct_mod, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass

    _ct_mod._RemainderColsList = _RemainderColsList


st.set_page_config(
    page_title="Visit with Us - Wellness Tourism Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package Purchase Prediction")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

try:
    model = joblib.load("models/best_model.pkl")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Customer Inputs
# ---------------------------------------------------------

st.sidebar.header("Customer Information")

Age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

CityTier = st.sidebar.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.sidebar.selectbox(
    "Occupation",
    ["Salaried", "Self Employed", "Small Business",
     "Large Business", "Free Lancer"]
)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.sidebar.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

PreferredPropertyStar = st.sidebar.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.sidebar.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=3
)

Passport = st.sidebar.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.sidebar.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.sidebar.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.sidebar.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.sidebar.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)

PitchSatisfactionScore = st.sidebar.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

ProductPitched = st.sidebar.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

NumberOfFollowups = st.sidebar.number_input(
    "Number of Followups",
    min_value=0,
    max_value=20,
    value=3
)

DurationOfPitch = st.sidebar.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=60,
    value=15
)

TypeofContact = st.sidebar.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)


# ---------------------------------------------------------
# Create input DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "ProductPitched": [ProductPitched],
    "NumberOfFollowups": [NumberOfFollowups],
    "DurationOfPitch": [DurationOfPitch]
})


# ---------------------------------------------------------
# Display customer information
# ---------------------------------------------------------

st.subheader("Customer Input")

st.dataframe(
    input_data,
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict Purchase", type="primary"):

    try:

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success(
                "🎉 Prediction: Customer is likely to purchase "
                "the Wellness Tourism Package."
            )
        else:
            st.warning(
                "Prediction: Customer is unlikely to purchase "
                "the Wellness Tourism Package."
            )

        # Display probability when available
        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)[0][1]

            st.metric(
                "Purchase Probability",
                f"{probability:.2%}"
            )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)
