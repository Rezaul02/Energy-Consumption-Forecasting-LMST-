import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔋 বিদ্যুৎ ব্যবহার পূর্বাভাস ")

# --- Constants ---
ELECTRICITY_RATE = 6.0  # Taka per kWh (user can change this)
PEAK_HOURS = [(18, 22)]  # 6PM-10PM as peak hours
OFF_PEAK_HOURS = [(0, 6)]  # 12AM-6AM as off-peak hours

# --- Bengali Energy Saving Tips ---
ENERGY_TIPS = {
    "high": [
        "এসি/ফ্যানের তাপমাত্রা ২৪-২৬°C এ সেট করুন",
        "অপ্রয়োজনীয় লাইট ও ইলেকট্রনিক্স বন্ধ করুন",
        "উচ্চ বিদ্যুৎখরচের যন্ত্রপাতি (ওভেন, ইস্ত্রি) অন্য সময়ে ব্যবহার করুন"
    ],
    "normal": [
        "LED বাল্ব ব্যবহার করুন",
        "যন্ত্রপাতি স্ট্যান্ডবাই মোডে না রেখে বন্ধ করুন",
        "প্রাকৃতিক আলো ব্যবহার করুন"
    ]
}

# --- Upload CSV File ---
df = pd.read_csv(r"C:\Users\Bulipe\Downloads\enhanced_dummy_energy_data.csv")

try:
    # --- Data Processing ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')

    if 'feels_like_C' not in df.columns and 'temp_C' in df.columns:
        df['feels_like_C'] = np.round(df['temp_C'] - 0.5 * np.random.rand(len(df)) * df['temp_C'], 1)

    if 'lag_1h_energy' not in df.columns and 'energy_kWh' in df.columns:
        df['lag_1h_energy'] = df['energy_kWh'].shift(1)
        df['lag_1h_energy'].fillna(df['energy_kWh'].iloc[0], inplace=True)

    df = df.dropna()

    X = df.drop(columns=["timestamp", "energy_kWh"])
    y = df["energy_kWh"]

    # --- Scaling ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))


    # --- Sequence Creation ---
    def create_sequences(X, y, time_steps=24):
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        return np.array(X_seq), np.array(y_seq)


    X_seq, y_seq = create_sequences(X_scaled, y_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # --- Train LSTM Model ---
    with st.spinner("মডেল ট্রেনিং চলছে..."):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    # --- Predict & Evaluate ---
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

    # --- Bengali Metrics Explanation ---
    st.subheader("📊 মডেল মূল্যায়ন মেট্রিক্স (বাংলায়)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE (গড় পরম ত্রুটি)", f"{mae:.2f} kWh")
        st.caption("""
        **ব্যাখ্যা:**  
        এটি আপনার পূর্বাভাস এবং আসল ব্যবহারের গড় পার্থক্য দেখায়।  
        উদাহরণ: MAE 1.5 kWh মানে আপনার পূর্বাভাস সাধারণত  
        আসল মান থেকে 1.5 kWh বেশি বা কম হবে।
        """)

    with col2:
        st.metric("RMSE (রুট মিন স্কোয়ার ত্রুটি)", f"{rmse:.2f} kWh")
        st.caption("""
        **ব্যাখ্যা:**  
        বড় ত্রুটিগুলিকে বেশি গুরুত্ব দিয়ে গড় ত্রুটি পরিমাপ করে।  
        RMSE যত কম, মডেল তত ভালো।
        """)

    with col3:
        st.metric("MAPE (গড় শতকরা পরম ত্রুটি)", f"{mape:.2f}%")
        st.caption("""
        **ব্যাখ্যা:**  
        শতকরা হিসাবে ত্রুটি দেখায়। 20% MAPE মানে পূর্বাভাস  
        সাধারণত আসল মান থেকে 20% ভুল হতে পারে।
        """)

    # --- Cost Conversion ---
    st.subheader("💰 বিদ্যুৎ খরচ হিসাব")
    rate = st.number_input("প্রতি kWh বিদ্যুতের দর (টাকায়)", min_value=0.0, value=6.0, step=0.5)

    total_actual_cost = np.sum(y_test_actual) * rate
    total_predicted_cost = np.sum(y_pred) * rate
    cost_difference = total_actual_cost - total_predicted_cost

    st.write(f"মোট প্রকৃত খরচ: {total_actual_cost:.2f} টাকা")
    st.write(f"মোট পূর্বাভাসিত খরচ: {total_predicted_cost:.2f} টাকা")
    if cost_difference > 0:
        st.warning(f"আপনি পূর্বাভাসের চেয়ে {cost_difference:.2f} টাকা বেশি খরচ করেছেন")
    else:
        st.success(f"আপনি পূর্বাভাসের চেয়ে {-cost_difference:.2f} টাকা কম খরচ করেছেন")

    # --- Anomaly Detection ---
    st.subheader("⚠️ অস্বাভাবিক ব্যবহার সতর্কতা")
    threshold = st.slider("অস্বাভাবিকতা সনাক্তকরণ থ্রেশহোল্ড (%)", 10, 50, 20)

    anomalies = np.where((y_test_actual - y_pred) / y_pred * 100 > threshold)[0]
    if len(anomalies) > 0:
        st.error(f"সতর্কতা: {len(anomalies)}টি সময়ে ব্যবহার পূর্বাভাসের চেয়ে {threshold}% বেশি ছিল")
        for idx in anomalies[:3]:  # Show first 3 anomalies
            st.write(f"- সময়: {idx}, পূর্বাভাস: {y_pred[idx][0]:.2f} kWh, আসল: {y_test_actual[idx][0]:.2f} kWh")
    else:
        st.success("কোন অস্বাভাবিক ব্যবহার পাওয়া যায়নি")

    # --- Time-based Usage Insights ---
    st.subheader("🕒 সময়ভিত্তিক বিদ্যুৎ ব্যবহার")

    # Add time information to test data
    test_dates = df.iloc[-len(y_test_actual):]['timestamp']
    df_test = pd.DataFrame({
        'timestamp': test_dates,
        'actual': y_test_actual.flatten(),
        'predicted': y_pred.flatten()
    })

    # Categorize by time of day
    df_test['hour'] = df_test['timestamp'].dt.hour
    df_test['time_category'] = 'normal'
    for start, end in PEAK_HOURS:
        df_test.loc[df_test['hour'].between(start, end), 'time_category'] = 'peak'
    for start, end in OFF_PEAK_HOURS:
        df_test.loc[df_test['hour'].between(start, end), 'time_category'] = 'off-peak'

    # Show average usage by time category
    time_stats = df_test.groupby('time_category')['actual'].mean()
    st.write("গড় বিদ্যুৎ ব্যবহার:")
    st.write(time_stats)

    # Advice based on time category
    st.write("**পরামর্শ:**")
    if 'peak' in time_stats.index:
        st.warning(
            f"পিক আওয়ারে (6PM-10PM) গড় ব্যবহার: {time_stats['peak']:.2f} kWh - এই সময়ে বিদ্যুৎ সবচেয়ে ব্যয়বহুল")
    if 'off-peak' in time_stats.index:
        st.success(f"অফ-পিক সময়ে (12AM-6AM) গড় ব্যবহার: {time_stats['off-peak']:.2f} kWh - এই সময়ে বিদ্যুৎ সস্তা")

    # --- Energy Saving Tips ---
    st.subheader("💡 বিদ্যুৎ সাশ্রয়ের টিপস")

    avg_usage = np.mean(y_test_actual)
    if avg_usage > np.mean(y_pred):
        tip_category = 'high'
    else:
        tip_category = 'normal'

    for tip in ENERGY_TIPS[tip_category]:
        st.write(f"- {tip}")

    # --- Plot Predictions ---
    st.subheader("Actual vs Predicted Energy Consumption")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot with anomaly markers
    ax.plot(y_test_actual, label="Actual", color='blue')
    ax.plot(y_pred, label="Predicted", color='red', linestyle='--')

    # Highlight anomalies
    if len(anomalies) > 0:
        ax.scatter(anomalies, y_test_actual[anomalies], color='orange', label="Unusual usage")

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Energy forcasting")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ ডেটা প্রসেসিংয়ে ত্রুটি: {e}")
else:
    st.warning("⚠️ শুরু করতে CSV ফাইল আপলোড করুন")