import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Enhanced DataViz Dashboard",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 Enhanced DataViz Dashboard")
st.markdown(
    """
    Explore, analyze, transform, visualize, and model your datasets interactively using Streamlit.
    """
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -----------------------------
# SIDEBAR MENU
# -----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dataset Overview",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Visualizations",
        "WordCloud",
        "Map Visualization",
        "Data Profiling",
        "Machine Learning"
    ]
)

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
if menu == "Dataset Overview":

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

# -----------------------------
# DATA CLEANING
# -----------------------------
elif menu == "Data Cleaning":

    st.subheader("Missing Value Analysis")

    st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        st.success("Missing numerical values filled using median strategy.")

    st.subheader("Duplicate Rows")

    duplicates = df.duplicated().sum()

    st.write(f"Duplicate rows found: {duplicates}")

    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        st.success("Duplicates removed successfully.")

# -----------------------------
# EDA
# -----------------------------
elif menu == "Exploratory Data Analysis":

    st.subheader("Statistical Summary")

    st.write(df.describe())

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

# -----------------------------
# VISUALIZATIONS
# -----------------------------
elif menu == "Visualizations":

    plot_type = st.selectbox(
        "Select Plot Type",
        [
            "Histogram",
            "Scatter Plot",
            "Box Plot",
            "Line Plot",
            "Pie Chart",
            "Area Plot"
        ]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    if plot_type == "Histogram":

        col = st.selectbox("Select Column", numeric_cols)

        fig = px.histogram(df, x=col)

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter Plot":

        x = st.selectbox("X Axis", all_cols)
        y = st.selectbox("Y Axis", numeric_cols)

        fig = px.scatter(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Box Plot":

        x = st.selectbox("Category", all_cols)
        y = st.selectbox("Value", numeric_cols)

        fig = px.box(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Line Plot":

        x = st.selectbox("X Axis", all_cols)
        y = st.selectbox("Y Axis", numeric_cols)

        fig = px.line(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Pie Chart":

        col = st.selectbox("Select Category", all_cols)

        fig = px.pie(df, names=col)

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Area Plot":

        x = st.selectbox("X Axis", all_cols)
        y = st.selectbox("Y Axis", numeric_cols)

        fig = px.area(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# WORD CLOUD
# -----------------------------
elif menu == "WordCloud":

    text_cols = df.select_dtypes(include="object").columns

    if len(text_cols) == 0:
        st.warning("No text columns found.")
    else:

        selected_col = st.selectbox(
            "Select Text Column",
            text_cols
        )

        text = " ".join(df[selected_col].astype(str))

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(15, 7))

        ax.imshow(wordcloud)

        ax.axis("off")

        st.pyplot(fig)

# -----------------------------
# MAP VISUALIZATION
# -----------------------------
elif menu == "Map Visualization":

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    lat_col = st.selectbox("Latitude Column", numeric_cols)
    lon_col = st.selectbox("Longitude Column", numeric_cols)

    if lat_col and lon_col:

        map_center = [
            df[lat_col].mean(),
            df[lon_col].mean()
        ]

        m = folium.Map(
            location=map_center,
            zoom_start=4
        )

        heat_data = list(
            zip(df[lat_col], df[lon_col])
        )

        HeatMap(heat_data).add_to(m)

        st_folium(m, width=900)

# -----------------------------
# DATA PROFILING
# -----------------------------
elif menu == "Data Profiling":

    st.subheader("Automated Data Profiling Report")

    profile = ProfileReport(
        df,
        explorative=True
    )

    profile_html = profile.to_html()

    components.html(
        profile_html,
        height=1000,
        scrolling=True
    )

# -----------------------------
# MACHINE LEARNING
# -----------------------------
elif menu == "Machine Learning":

    st.subheader("ML Model Training")

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    features = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target]
    )

    model_name = st.selectbox(
        "Select Model",
        [
            "Random Forest",
            "Logistic Regression"
        ]
    )

    if st.button("Train Model"):

        if len(features) == 0:
            st.warning("Please select feature columns.")
            st.stop()

        model_df = df[features + [target]].copy()

        for col in model_df.columns:

            if model_df[col].dtype == "object":

                encoder = LabelEncoder()

                model_df[col] = encoder.fit_transform(
                    model_df[col].astype(str)
                )

        X = model_df[features]
        y = model_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        if model_name == "Random Forest":

            model = RandomForestClassifier()

        else:

            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        st.success(f"Accuracy: {accuracy:.2f}")

        st.subheader("Classification Report")

        st.text(
            classification_report(
                y_test,
                predictions
            )
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)