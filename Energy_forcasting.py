# %% [markdown]
# # This project is Energy Consumasion forcusting 

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configs ---
DAYS = 30  # 30 days of data
HOURS = 24 * DAYS
np.random.seed(42)

# --- 1. Generate Timestamps ---
timestamps = pd.date_range("2024-01-01", periods=HOURS, freq="H")

# --- 2. Enhanced Energy (kWh) with Outliers ---
base_energy = 10
hour_of_day = timestamps.hour.to_numpy()
day_of_week = timestamps.dayofweek.to_numpy()

energy = (
    base_energy
    + 5 * np.sin(hour_of_day * np.pi / 12)           # Daily pattern
    - 2 * (day_of_week >= 5)                         # Weekend energy drop
    + np.random.normal(0, 1.5, HOURS)                # Noise
)
energy = np.abs(energy)

# Add outliers (1% of data points)
outlier_mask = np.random.binomial(1, 0.01, HOURS).astype(bool)
energy[outlier_mask] *= 2.5  # Spike outliers

# --- 3. Weather Data with Missing Values ---
temp = 20 + 10 * np.sin(hour_of_day * np.pi / 12)
temp += np.random.normal(0, 2, HOURS)

# Add missing weather data (5%)
missing_mask = np.random.binomial(1, 0.05, HOURS).astype(bool)
temp = np.where(missing_mask, np.nan, temp)

# --- 4. Occupancy with Sensor Errors ---
occupancy = np.zeros(HOURS)
for i, hour in enumerate(hour_of_day):
    if 8 <= hour <= 18:
        occupancy[i] = np.random.randint(20, 30) if 12 <= hour <= 13 else np.random.randint(40, 60)
    else:
        occupancy[i] = np.random.randint(0, 5)

# Simulate sensor failures (2%)
sensor_failure = np.random.rand(HOURS) < 0.02
occupancy = np.where(sensor_failure, 0, occupancy)

# --- 5. Combine into DataFrame ---
df = pd.DataFrame({
    "timestamp": timestamps,
    "energy_kWh": np.round(energy, 2),
    "temp_C": np.round(temp, 1),
    "feels_like_C": np.round(temp - 0.5 * np.random.rand(HOURS) * temp, 1),
    "precip": np.random.choice([0, 1, 2], HOURS, p=[0.7, 0.2, 0.1]),
    "humidity_pct": np.clip(50 + temp - 20 + np.random.normal(0, 5, HOURS), 30, 80),
    "occupancy": occupancy,
    "is_weekend": (day_of_week >= 5).astype(int),
    "is_holiday": np.random.binomial(1, 0.02, HOURS),
    "lag_1h_energy": np.round(np.roll(energy, 1), 2)
})
df.loc[0, 'lag_1h_energy'] = df.loc[0, 'energy_kWh']  # Fix first row value

# --- 6. Add Missing Energy Readings (3%) ---
df.loc[df.sample(frac=0.03).index, 'energy_kWh'] = np.nan

# --- 7. Save and Show Sample ---
df.to_csv("enhanced_dummy_energy_data.csv", index=False)
df.head(5)


# %%
df.isnull().sum()

# %%


# %%
# --- Handle Missing energy_kWh ---
df['energy_kWh'] = df['energy_kWh'].fillna(method='ffill')  # Forward fill
df['energy_kWh'] = df['energy_kWh'].fillna(method='bfill')  # Backup if first value is NaN

# --- Interpolate temp_C ---
df['temp_C'] = df['temp_C'].interpolate(method='linear', limit_direction='both')

# --- Recalculate feels_like_C using updated temp_C ---
df['feels_like_C'] = np.round(df['temp_C'] - 0.5 * np.random.rand(len(df)) * df['temp_C'], 1)

# --- Recalculate humidity_pct based on temp_C ---
df['humidity_pct'] = np.clip(50 + df['temp_C'] - 20 + np.random.normal(0, 5, len(df)), 30, 80)


# %%
df.isnull().sum()

# %% [markdown]
#  #  Outlier is present or not 

# %%
def handle_outliers_iqr(series):
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)


# %%
def handle_quantial(seris): 
    Q1=seris.quantile(0.25)
    Q2=seris.quantile(.75)
    iqr=Q3-Q1
    lower=Q1-1.5*iqr 
    upper =Q3-1.5*iqr 
    return seris.clip(lower , upper)


# %%
# Apply to energy_kWh
df['energy_kWh'] = handle_outliers_iqr(df['energy_kWh'])

# Apply to temperature
df['temp_C'] = handle_outliers_iqr(df['temp_C'])

# Apply to feels_like_C
df['feels_like_C'] = handle_outliers_iqr(df['feels_like_C'])

# Apply to humidity_pct
df['humidity_pct'] = handle_outliers_iqr(df['humidity_pct'])

# (Optional) Apply to occupancy if it has large spikes
df['occupancy'] = handle_outliers_iqr(df['occupancy'])


# %%
import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = ['energy_kWh', 'temp_C', 'feels_like_C', 'humidity_pct', 'occupancy']

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

for col in numeric_cols:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers detected")
    plt.figure(figsize=(8, 4))
    sns.kdeplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# %%
def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

df = remove_outliers_iqr(df, 'energy_kWh')
df = remove_outliers_iqr(df, 'temp_C')
df=remove_outliers_iqr(df, "feels_like_C")
df = remove_outliers_iqr(df, 'humidity_pct')
df = remove_outliers_iqr(df, 'occupancy')

# %%
import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = ['energy_kWh', 'temp_C', 'feels_like_C', 'humidity_pct', 'occupancy']

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

for col in numeric_cols:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers detected")
    plt.figure(figsize=(8, 4))
    sns.kdeplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# %% [markdown]
# # Normalization part start / Satrandlization 

# %%
from sklearn.preprocessing import StandardScaler
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# %%
df

# %%
df.dtypes

# %%
df.size

# %% [markdown]
# # Now encodding technique is below that  which is not important for my dataset becaue my dataset almost int and float datatyper 

# %%
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# %% [markdown]
# # Time serise Analaysis 

# %%
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['energy_kWh'], label='Energy Consumption')
plt.title('Energy Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Energy (kWh)')
plt.grid(True)
plt.legend()
plt.show()

# %%
df.shape[0]

# %%
df.shape[1]

# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Convert the datetime column to datetime format and set it as index
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Replace 'timestamp' with your actual datetime column name
df.set_index('timestamp', inplace=True)

# Resample to daily data (mean per day)
daily_energy = df['energy_kWh'].resample('D').mean()

# Decompose the time series
decomposition = seasonal_decompose(daily_energy, model='additive', period=7)  # Weekly seasonality

# Plot the decomposition
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(daily_energy, label='Original')
plt.legend(loc='upper left')
plt.title('Energy Consumption Decomposition')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# %%
# Extract day of week and hour of day for analysis
df['day_of_week'] = df.index.dayofweek
df['hour_of_day'] = df.index.hour

# Plot average energy by hour of day
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour_of_day', y='energy_kWh', data=df)
plt.title('Average Energy Consumption by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Energy (kWh)')
plt.grid(True)
plt.show()

# Plot average energy by day of week
plt.figure(figsize=(12, 6))
sns.lineplot(x='day_of_week', y='energy_kWh', data=df)
plt.title('Average Energy Consumption by Day of Week')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Average Energy (kWh)')
plt.grid(True)
plt.show()

# %% [markdown]
# # Co-Relation martix 

# %%
# Select numerical columns for correlation analysis
numerical_cols = ['energy_kWh', 'temp_C', 'occupancy', 'humidity_pct', 'precip', 'lag_1h_energy']
corr_matrix = df[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Energy and Other Variables')
plt.show()

# %%
# Energy vs Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp_C', y='energy_kWh', data=df)
plt.title('Energy Consumption vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy (kWh)')
plt.grid(True)
plt.show()

# Energy vs Occupancy
plt.figure(figsize=(10, 6))
sns.scatterplot(x='occupancy', y='energy_kWh', data=df)
plt.title('Energy Consumption vs Occupancy')
plt.xlabel('Occupancy')
plt.ylabel('Energy (kWh)')
plt.grid(True)
plt.show()

# Energy vs Lagged Energy
plt.figure(figsize=(10, 6))
sns.scatterplot(x='lag_1h_energy', y='energy_kWh', data=df)
plt.title('Current Energy vs Lagged (1h) Energy')
plt.xlabel('Lagged Energy (kWh)')
plt.ylabel('Current Energy (kWh)')
plt.grid(True)
plt.show()

# %% [markdown]
# # Model devlopment 

# %%
df.dtypes


# %%
X = df.drop(columns=["energy_kWh"])  # Features (all columns except target)
y = df["energy_kWh"]  

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

# Separate numerical features (excluding timestamp if present)
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X_num = X[numerical_cols].values
y = y.values.reshape(-1, 1)  # Reshape for scaling

# Scale features (X) and target (y)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_num)  # Normalized features
y_scaled = scaler_y.fit_transform(y)      # Normalized target

# %%
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])  # Past 'time_steps' hours
        y_seq.append(y[i+time_steps])    # Next hour's energy
    return np.array(X_seq), np.array(y_seq)

time_steps = 24  # Use 24 hours to predict the next hour
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# %%
# Predictions (scaled)
y_pred_scaled = model.predict(X_test)

# Inverse scaling to get actual kWh values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"RMSE: {rmse:.2f} kWh")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Energy')
plt.plot(y_pred, label='Predicted Energy')
plt.xlabel('Time Steps')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.show()

# %% [markdown]
# # Evaulate this model 

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Inverse transform if using StandardScaler/MinMaxScaler
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate MAE, RMSE, MAPE
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

print(f"MAE: {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"MAPE: {mape:.2f}%")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Energy (kWh)', color='blue')
plt.plot(y_pred_actual, label='Predicted Energy (kWh)', color='red', linestyle='--')
plt.title("Actual vs. Predicted Energy Consumption")
plt.xlabel("Time Steps")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



