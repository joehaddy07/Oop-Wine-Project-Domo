# ============================================================
# STREAMLIT WINE QUALITY PREDICTION APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🍷 Wine Quality Prediction App")

st.write(
    "Enter the characteristics of a wine below "
    "to predict whether it is good or bad quality."
)


# ============================================================
# LOAD THE TRAINED MODEL
# ============================================================

# Get the directory where this app.py file is located.
APP_DIR = Path(__file__).resolve().parent

# Build the path to the trained model.
MODEL_PATH = APP_DIR / "wine_quality_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained machine-learning model.

    @st.cache_resource tells Streamlit to load the model
    once instead of loading it every time the application
    reruns.
    """

    return joblib.load(MODEL_PATH)


# Load the model.
model = load_model()


# ============================================================
# WINE INPUT SECTION
# ============================================================

st.subheader("Enter Wine Characteristics")


# Create two columns to make the interface cleaner.
col1, col2 = st.columns(2)


with col1:

    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=0.0,
        value=7.4
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.0,
        value=0.7
    )

    citric_acid = st.number_input(
        "Citric Acid",
        min_value=0.0,
        value=0.0
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        min_value=0.0,
        value=1.9
    )

    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        value=0.076
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        min_value=0.0,
        value=11.0
    )


with col2:

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        min_value=0.0,
        value=34.0
    )

    density = st.number_input(
        "Density",
        min_value=0.0,
        value=0.9978
    )

    pH = st.number_input(
        "pH",
        min_value=0.0,
        value=3.51
    )

    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        value=0.56
    )

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=9.4
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🍷 Predict Quality",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create a dictionary containing the 11 features.
    # --------------------------------------------------------

    wine_data = {

        "fixed acidity": fixed_acidity,

        "volatile acidity": volatile_acidity,

        "citric acid": citric_acid,

        "residual sugar": residual_sugar,

        "chlorides": chlorides,

        "free sulfur dioxide": free_sulfur_dioxide,

        "total sulfur dioxide": total_sulfur_dioxide,

        "density": density,

        "pH": pH,

        "sulphates": sulphates,

        "alcohol": alcohol
    }


    # --------------------------------------------------------
    # Convert the dictionary into a DataFrame.
    #
    # The trained model expects a DataFrame with the same
    # feature names used during training.
    # --------------------------------------------------------

    wine_df = pd.DataFrame([wine_data])


    # --------------------------------------------------------
    # Make the prediction.
    # --------------------------------------------------------

    prediction = model.predict(wine_df)


    # --------------------------------------------------------
    # Display the result.
    # --------------------------------------------------------

    if prediction[0] == 1:

        st.success(
            "🍷 Good Quality Wine"
        )

    else:

        st.error(
            "🍷 Bad Quality Wine"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About")

st.sidebar.write(
    """
    This application uses a trained machine-learning model
    to classify wine quality.

    The model uses 11 wine characteristics as input.
    """
)