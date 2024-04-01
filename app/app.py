import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import streamlit as st


from functions import (
    load_models,
    plot_preds,
    prediction_result,
    user_data_prep,
    user_input_features,
)


st.set_page_config(
    page_title="Model Demo",
    page_icon="🚀",
)

LOGO = "./logo/vegas.png"

# Title
st.title("Food Inspection Model")

# Subheader
st.subheader("_Adjust the inputs in the sidebar to get predictions_")

# Logo
st.sidebar.image(LOGO, use_column_width=True)
st.sidebar.text("Jason Lee\nLead Data Scientist")


# Sidebar User Inputs
input_df = user_input_features()

# load models
model = load_models()

# Prediction
user_prepped = user_data_prep(input_df)
prediction = prediction_result(model, user_prepped)
st.subheader("Prediction:")
fig = plot_preds(prediction)
st.plotly_chart(fig)
st.dataframe(prediction)
