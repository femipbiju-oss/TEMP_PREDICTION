import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("temperature_rnn.keras")

# Title
st.title("Next Temperature Prediction")

st.write(
    "Enter the temperature and vibration values from the previous "
    "two timestamps to predict the next temperature."
)

# Timestamp 1
st.subheader("Timestamp 1")

temperature_1 = st.number_input(
    "Temperature 1",
    value=46.0
)

vibration_1 = st.number_input(
    "Vibration 1",
    value=2.1
)

# Timestamp 2
st.subheader("Timestamp 2")

temperature_2 = st.number_input(
    "Temperature 2",
    value=48.0
)

vibration_2 = st.number_input(
    "Vibration 2",
    value=2.3
)

# Prediction button
if st.button("Predict Next Temperature"):

    # Create input in RNN format:
    # (samples, time steps, features)
    new_temperature = np.array([
        [
            [temperature_1, vibration_1],
            [temperature_2, vibration_2]
        ]
    ])

    # Predict
    next_temperature = model.predict(
        new_temperature,
        verbose=0
    )

    # Display result
    st.success(
        f"Predicted Next Temperature: "
        f"{float(next_temperature[0][0]):.2f}"
    )
