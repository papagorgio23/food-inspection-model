import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


categories = [
    "Restaurant",
    "Snack Bar",
    "Institutional Food Service",
    "Meat/Poultry/Seafood",
    "Grocery Store Sampling",
    "Bar / Tavern",
    "Food Trucks / Mobile Vendor",
    "Portable Unit",
    "Caterer",
    "Special Kitchen",
    "Frozen Meat Sales",
    "Farmer's Market",
    "Childcare Kitchens",
    "Buffet",
    "Produce Market",
    "Elementary School Kitchen",
    "Bakery Sales",
    "Concessions",
    "Confection",
    "Kitchen Bakery",
    "Banquet Kitchen",
    "Pantry",
    "Self-Service Food Truck",
    "Barbeque",
    "Portable Bar",
    "Main Kitchen",
    "Vegetable Prep",
]


@st.cache_data()
def load_models():
    # load best saved model
    best_model = joblib.load("models/model.pkl")
    return best_model


def user_input_features():

    inspection_type = st.sidebar.selectbox(
        "Inspection_Type", ("Routine Inspection", "Re-inspection", "Survey"), index=0
    )
    days_since_previous_inspection = st.sidebar.slider(
        "Days Since Last Inspection", min_value=0, max_value=730, value=100
    )
    is_first_inspection = st.sidebar.selectbox(
        "Is this the First Inspection?", ("Yes", "No"), index=1
    )
    Category_Name = st.sidebar.selectbox("Category_Name", (categories))
    Current_Grade = st.sidebar.selectbox("Current_Grade", ("A", "B", "C", "X"), index=0)
    Current_Demerits = st.sidebar.slider(
        "Current Demerits", min_value=0, max_value=40, value=3
    )
    previous_demerits = st.sidebar.slider(
        "Previous Demerits", min_value=0, max_value=40, value=3
    )

    data = {
        "days_since_previous_inspection": [days_since_previous_inspection],
        "Inspection_Type": [inspection_type],
        "is_first_inspection": [is_first_inspection],
        "previous_demerits": [previous_demerits],
        "Category_Name": [Category_Name],
        "Current_Grade": [Current_Grade],
        "Current_Demerits": [Current_Demerits],
    }

    features = pd.DataFrame(data)

    return features


# function to perform all the data cleaning/feature engineering
def user_data_prep(df: pd.DataFrame) -> pd.DataFrame:

    final_df = pd.DataFrame(
        np.zeros((1, 38), dtype="int64"),
        columns=[
            "days_since_previous_inspection",
            "is_first_inspection",
            "previous_demerits",
            "Current_Demerits",
            "Inspection_Type_Re-inspection",
            "Inspection_Type_Routine Inspection",
            "Inspection_Type_Survey",
            "Category_Name_Bakery Sales",
            "Category_Name_Banquet Kitchen",
            "Category_Name_Bar / Tavern",
            "Category_Name_Barbeque",
            "Category_Name_Buffet",
            "Category_Name_Caterer",
            "Category_Name_Childcare Kitchens",
            "Category_Name_Concessions",
            "Category_Name_Confection",
            "Category_Name_Elementary School Kitchen",
            "Category_Name_Farmer's Market",
            "Category_Name_Food Trucks / Mobile Vendor",
            "Category_Name_Frozen Meat Sales",
            "Category_Name_Grocery Store Sampling",
            "Category_Name_Institutional Food Service",
            "Category_Name_Kitchen Bakery",
            "Category_Name_Main Kitchen",
            "Category_Name_Meat/Poultry/Seafood",
            "Category_Name_Pantry",
            "Category_Name_Portable Bar",
            "Category_Name_Portable Unit",
            "Category_Name_Produce Market",
            "Category_Name_Restaurant",
            "Category_Name_Self-Service Food Truck",
            "Category_Name_Snack Bar",
            "Category_Name_Special Kitchen",
            "Category_Name_Vegetable Prep",
            "Current_Grade_A",
            "Current_Grade_B",
            "Current_Grade_C",
            "Current_Grade_X",
        ],
    )

    # Days since previous inspection
    final_df["days_since_previous_inspection"] = df["days_since_previous_inspection"][0]

    # First Inspection
    final_df["is_first_inspection"] = np.where(
        df["is_first_inspection"][0] == "Yes", 1, 0
    )

    # Previous Demerits
    final_df["previous_demerits"] = df["previous_demerits"][0]
    # Current Demerits
    final_df["Current_Demerits"] = df["Current_Demerits"][0]

    # Inspection Type
    final_df["Inspection_Type_Re-inspection"] = np.where(
        df["Inspection_Type"][0] == "Re-inspection", 1, 0
    )
    final_df["Inspection_Type_Routine Inspection"] = np.where(
        df["Inspection_Type"][0] == "Routine Inspection", 1, 0
    )
    final_df["Inspection_Type_Survey"] = np.where(
        df["Inspection_Type"][0] == "Survey", 1, 0
    )

    # Category Name
    final_df["Category_Name_Bakery Sales"] = np.where(
        df["Category_Name"][0] == "Bakery Sales", 1, 0
    )
    final_df["Category_Name_Banquet Kitchen"] = np.where(
        df["Category_Name"][0] == "Banquet Kitchen", 1, 0
    )
    final_df["Category_Name_Bar / Tavern"] = np.where(
        df["Category_Name"][0] == "Bar / Tavern", 1, 0
    )
    final_df["Category_Name_Barbeque"] = np.where(
        df["Category_Name"][0] == "Barbeque", 1, 0
    )
    final_df["Category_Name_Buffet"] = np.where(
        df["Category_Name"][0] == "Buffet", 1, 0
    )
    final_df["Category_Name_Caterer"] = np.where(
        df["Category_Name"][0] == "Caterer", 1, 0
    )
    final_df["Category_Name_Childcare Kitchens"] = np.where(
        df["Category_Name"][0] == "Childcare Kitchens", 1, 0
    )
    final_df["Category_Name_Concessions"] = np.where(
        df["Category_Name"][0] == "Concessions", 1, 0
    )
    final_df["Category_Name_Confection"] = np.where(
        df["Category_Name"][0] == "Confection", 1, 0
    )
    final_df["Category_Name_Elementary School Kitchen"] = np.where(
        df["Category_Name"][0] == "Elementary School Kitchen", 1, 0
    )
    final_df["Category_Name_Farmer's Market"] = np.where(
        df["Category_Name"][0] == "Farmer's Market", 1, 0
    )
    final_df["Category_Name_Food Trucks / Mobile Vendor"] = np.where(
        df["Category_Name"][0] == "Food Trucks / Mobile Vendor", 1, 0
    )
    final_df["Category_Name_Frozen Meat Sales"] = np.where(
        df["Category_Name"][0] == "Frozen Meat Sales", 1, 0
    )
    final_df["Category_Name_Grocery Store Sampling"] = np.where(
        df["Category_Name"][0] == "Grocery Store Sampling", 1, 0
    )
    final_df["Category_Name_Institutional Food Service"] = np.where(
        df["Category_Name"][0] == "Institutional Food Service", 1, 0
    )
    final_df["Category_Name_Kitchen Bakery"] = np.where(
        df["Category_Name"][0] == "Kitchen Bakery", 1, 0
    )
    final_df["Category_Name_Main Kitchen"] = np.where(
        df["Category_Name"][0] == "Main Kitchen", 1, 0
    )
    final_df["Category_Name_Meat/Poultry/Seafood"] = np.where(
        df["Category_Name"][0] == "Meat/Poultry/Seafood", 1, 0
    )
    final_df["Category_Name_Pantry"] = np.where(
        df["Category_Name"][0] == "Pantry", 1, 0
    )
    final_df["Category_Name_Portable Bar"] = np.where(
        df["Category_Name"][0] == "Portable Bar", 1, 0
    )
    final_df["Category_Name_Portable Unit"] = np.where(
        df["Category_Name"][0] == "Portable Unit", 1, 0
    )
    final_df["Category_Name_Produce Market"] = np.where(
        df["Category_Name"][0] == "Produce Market", 1, 0
    )
    final_df["Category_Name_Restaurant"] = np.where(
        df["Category_Name"][0] == "Restaurant", 1, 0
    )
    final_df["Category_Name_Self-Service Food Truck"] = np.where(
        df["Category_Name"][0] == "Self-Service Food Truck", 1, 0
    )
    final_df["Category_Name_Snack Bar"] = np.where(
        df["Category_Name"][0] == "Snack Bar", 1, 0
    )
    final_df["Category_Name_Special Kitchen"] = np.where(
        df["Category_Name"][0] == "Special Kitchen", 1, 0
    )
    final_df["Category_Name_Vegetable Prep"] = np.where(
        df["Category_Name"][0] == "Vegetable Prep", 1, 0
    )

    # Current Grade
    final_df["Current_Grade_A"] = np.where(df["Current_Grade"][0] == "A", 1, 0)
    final_df["Current_Grade_B"] = np.where(df["Current_Grade"][0] == "B", 1, 0)
    final_df["Current_Grade_C"] = np.where(df["Current_Grade"][0] == "C", 1, 0)
    final_df["Current_Grade_X"] = np.where(df["Current_Grade"][0] == "X", 1, 0)

    return final_df


def prediction_result(model, data) -> pd.DataFrame:
    preds_prob = model.predict_proba(data)
    labels = ["Fail", "Pass"]

    new_pred = pd.DataFrame(preds_prob, columns=labels)
    new_pred = new_pred[["Pass", "Fail"]]

    return new_pred


def plot_preds(prediction):
    # plot preds with plotly

    # reshape data
    prediction = prediction.T.reset_index()
    prediction.columns = ["Prediction", "Probability"]

    fig = px.bar(
        prediction,
        x="Prediction",
        y="Probability",
        color="Prediction",
    )
    fig.update_layout(
        title="Food Inspection Model Prediction",
        xaxis_title="Prediction",
        yaxis_title="Probability",
    )
    # make y label percentage
    fig.update_yaxes(tickformat=".0%")

    return fig
