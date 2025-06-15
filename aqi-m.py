# ----------------------
# 1. Import Libraries
# ----------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ----------------------
# 2. Load Dataset
# ----------------------
df = pd.read_csv("GlobalAirQuality.csv")
print("Initial shape:", df.shape)

# ----------------------
# 3. AQI Calculation based on PM2.5 (EPA formula)
# ----------------------
def calculate_aqi_pm25(pm):
    if pd.isna(pm): return np.nan
    if pm <= 12.0:
        return (50 / 12.0) * pm
    elif pm <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm - 12.1) + 51
    elif pm <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm - 35.5) + 101
    elif pm <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm - 55.5) + 151
    elif pm <= 250.4:
        return ((300 - 201) / (250.4 - 150.5)) * (pm - 150.5) + 201
    elif pm <= 350.4:
        return ((400 - 301) / (350.4 - 250.5)) * (pm - 250.5) + 301
    elif pm <= 500.4:
        return ((500 - 401) / (500.4 - 350.5)) * (pm - 350.5) + 401
    else:
        return 500

df['AQI'] = df['PM2.5'].apply(calculate_aqi_pm25)

# ----------------------
# 4. Data Cleaning
# ----------------------
threshold = len(df) * 0.5
df.dropna(axis=1, thresh=threshold, inplace=True)

# Fill numeric with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------
# 5. Feature Engineering: Date
# ----------------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year.fillna(0).astype(int)
    df['Month'] = df['Date'].dt.month.fillna(0).astype(int)
    df['Day'] = df['Date'].dt.day.fillna(0).astype(int)

# ----------------------
# 6. Keep City and Country Names + Encode
# ----------------------
df['City_Label'] = df['City']
df['Country_Label'] = df['Country']

city_encoder = LabelEncoder()
country_encoder = LabelEncoder()

if 'City' in df.columns:
    df['City'] = city_encoder.fit_transform(df['City'].astype(str))
    city_label_to_code = dict(zip(city_encoder.classes_, city_encoder.transform(city_encoder.classes_)))
    city_code_to_label = dict(zip(city_encoder.transform(city_encoder.classes_), city_encoder.classes_))

if 'Country' in df.columns:
    df['Country'] = country_encoder.fit_transform(df['Country'].astype(str))
    country_label_to_code = dict(zip(country_encoder.classes_, country_encoder.transform(country_encoder.classes_)))
    country_code_to_label = dict(zip(country_encoder.transform(country_encoder.classes_), country_encoder.classes_))

# ----------------------
# 7. Summary
# ----------------------
print("\nSummary Statistics:")
print(df.describe())

# ----------------------
# 8. Correlation
# ----------------------
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ----------------------
# 9. Distributions
# ----------------------
for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']:
    if col in df.columns:
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f"{col} Distribution")
        plt.show()

        sns.boxplot(y=df[col])
        plt.title(f"{col} Boxplot")
        plt.show()

# ----------------------
# 10. Save Cleaned Dataset
# ----------------------
df.to_csv("GlobalAirQuality_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'GlobalAirQuality_cleaned.csv'")

# ----------------------
# 11. Train Model on Full Feature Set
# ----------------------
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
            'Temperature', 'Humidity', 'City', 'Country', 'Year', 'Month', 'Day']
features = [f for f in features if f in df.columns]
df = df.dropna(subset=features + ['AQI'])

X = df[features]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")
