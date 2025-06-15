import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date

# --- Page config ---
st.set_page_config(page_title="🌍 AQI Predictor", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f3f9fd;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        }
        h1, h2, h3 {
            color: #003366;
        }
        .css-1d391kg, .css-1vq4p4l {
            background-color: #d9f4ff !important;
        }
        .css-1d391kg h2 {
            color: #004d66;
        }
        .stButton > button {
            background-color: #009999;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            transition: background-color 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #007777;
        }
        .stSlider, .stSelectbox, .stDateInput {
            margin-bottom: 1rem !important;
        }
        .stAlert-success {
            background-color: #e0f7fa;
            border-left: 6px solid #009999;
        }
        .stDataFrame {
            border-radius: 10px;
            border: 1px solid #dddddd;
        }
        h1 {
            background: linear-gradient(to right, #003366, #009999);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("GlobalAirQuality_cleaned.csv")
    city_map = dict(zip(df['City_Label'], df['City']))
    country_map = dict(zip(df['Country_Label'], df['Country']))
    return df, city_map, country_map

df, city_label_to_code, country_label_to_code = load_data()

# --- Feature Selection ---
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
            'Temperature', 'Humidity', 'City', 'Country', 'Year', 'Month', 'Day']
features = [f for f in features if f in df.columns]

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Choose section:", ["Introduction", "EDA", "Model", "Conclusion"])

# --- Introduction ---
if section == "Introduction":
    st.title("🌍 Air Quality Index (AQI) Predictor")
    st.markdown("""
    Welcome to the **AQI Predictor App**!  
    This interactive dashboard allows you to:
    
    - 📊 Explore global air quality trends  
    - 🧪 Analyze pollution levels in top cities  
    - 🤖 Predict AQI values using Machine Learning  
    - 🧠 Learn insights through visual storytelling
    """)

# --- EDA Section ---
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    
    st.subheader("🔢 Summary Statistics")
    st.write(df.describe())

    st.subheader("🔥 Correlation Heatmap")
    corr = df[features + ["AQI"]].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax, linewidths=0.5)
    st.pyplot(fig)

    st.subheader("📈 AQI Trend Over Time")
    time_trend = df.groupby(['Year', 'Month'])['AQI'].mean().reset_index()
    time_trend['Date'] = pd.to_datetime(time_trend[['Year', 'Month']].assign(DAY=1))
    fig, ax = plt.subplots()
    sns.lineplot(data=time_trend, x='Date', y='AQI', marker='o', color='teal')
    ax.set_title("Average AQI Over Time", fontsize=14)
    ax.set_ylabel("AQI")
    st.pyplot(fig)

    st.subheader("🏙️ Top 10 Polluted Cities")
    top_cities = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_cities.values, y=top_cities.index, palette='Reds_r', ax=ax)
    ax.set_xlabel("Average AQI")
    ax.set_ylabel("City")
    ax.tick_params(axis='y', labelsize=10)
    st.pyplot(fig)

    st.subheader("🌎 Top Polluted Countries")
    top_countries = df.groupby("Country")["AQI"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='Oranges_r', ax=ax)
    ax.set_xlabel("Average AQI")
    ax.set_ylabel("Country")
    ax.tick_params(axis='y', labelsize=10)
    st.pyplot(fig)

    st.subheader("📦 Feature Distributions")
    num_cols = 2
    for i in range(0, len(features), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(features):
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.histplot(df[features[i + j]], kde=True, color='skyblue', ax=ax)
                    ax.set_title(f"{features[i + j]} Distribution")
                    st.pyplot(fig)

# --- Model Section ---
elif section == "Model":
    st.header("🤖 AQI Prediction Model")

    df.dropna(subset=features + ['AQI'], inplace=True)
    X = df[features]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("🎯 Predict AQI from Your Input")

    input_data = []
    st.markdown("#### 📥 Pollutant and Weather Inputs")
    pollutant_cols = st.columns(4)
    for idx, feat in enumerate(['PM2.5', 'PM10', 'NO2', 'SO2']):
        with pollutant_cols[idx]:
            val = st.slider(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
            input_data.append(val)

    pollutant_cols_2 = st.columns(4)
    for idx, feat in enumerate(['CO', 'O3', 'Temperature', 'Humidity']):
        with pollutant_cols_2[idx]:
            val = st.slider(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
            input_data.append(val)

    st.markdown("#### 🗺️ Location and Date")
    loc1, loc2, loc3 = st.columns(3)
    with loc1:
        selected_city = st.selectbox("City", list(city_label_to_code.keys()))
    with loc2:
        selected_country = st.selectbox("Country", list(country_label_to_code.keys()))
    with loc3:
        selected_date = st.date_input("Date", date.today())

    input_data.append(city_label_to_code[selected_city])
    input_data.append(country_label_to_code[selected_country])
    input_data.append(selected_date.year)
    input_data.append(selected_date.month)
    input_data.append(selected_date.day)

    if st.button("🚀 Predict AQI"):
        input_df = pd.DataFrame([input_data], columns=features)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"🌫️ **Predicted AQI: {prediction[0]:.2f}**")

# --- Conclusion ---
elif section == "Conclusion":
    st.header("📝 Conclusion")
    st.markdown("""
    ### Summary
    - ✅ **PM2.5** values were used to compute AQI using **EPA standards**.
    - 🧠 A **Random Forest** model was trained with pollutants, weather, location, and date.
    - 📊 **EDA** revealed temporal trends and top contributors to air pollution.
    - 🧩 You can further enhance this app with **maps**, **alerts**, or **forecasting tools**.
    """)
