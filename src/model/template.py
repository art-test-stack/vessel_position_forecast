# 1. Preprocessing the training data

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your training data (example format: ['vesselId', 'timestamp', 'latitude', 'longitude', 'speed', ...])
# Assume df_train is your training DataFrame

# Sort by vesselId and timestamp
df_train = df_train.sort_values(by=['vesselId', 'timestamp'])

# Scaling the features (if necessary)
scaler = StandardScaler()
features = ['latitude', 'longitude', 'speed', ...]  # your feature list
df_train[features] = scaler.fit_transform(df_train[features])

# Prepare data for LSTM - group by vesselId
grouped = df_train.groupby('vesselId')

# Function to prepare data for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length].values
        target = data[i+sequence_length].values
        sequences.append(seq)
        target Ss.append(target)
    return sequences, targets

sequence_length = 10  # or any desired sequence length
X, y = [], []

for vessel_id, group in grouped:
    sequences, targets = create_sequences(group[features], sequence_length)
    X.extend(sequences)
    y.extend(targets)

import numpy as np
X = np.array(X)
y = np.array(y)

# Split data into training and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 2. Model Definition (LSTM here)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define an LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(sequence_length, len(features))))
model.add(Dense(len(features)))  # Output size equal to the number of features
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))


# 3. Prediction (Iterative Forecasting)

def iterative_forecast(last_known, model, steps, sequence_length):
    predicted = []
    current_sequence = last_known[-sequence_length:]

    for _ in range(steps):
        next_pred = model.predict(current_sequence.reshape(1, sequence_length, -1))[0]
        predicted.append(next_pred)
        
        # Update current_sequence by appending next prediction
        current_sequence = np.append(current_sequence[1:], [next_pred], axis=0)
    
    return predicted

# Let's assume you have a test DataFrame (df_test) with ['vesselId', 'timestamp']
df_test = df_test.sort_values(by=['vesselId', 'timestamp'])
grouped_test = df_test.groupby('vesselId')

# Make predictions for each vessel
forecast_steps = 5  # how many steps into the future you want to forecast
predictions = {}

for vessel_id, group in grouped_test:
    last_known_features = scaler.transform(group[features].values[-sequence_length:])  # Get the last known sequence
    future_preds = iterative_forecast(last_known_features, model, forecast_steps, sequence_length)
    
    # Store the predictions
    predictions[vessel_id] = scaler.inverse_transform(future_preds)  # Re-scale the predictions back to original

# Output predictions
for vessel_id, preds in predictions.items():
    print(f"Vessel {vessel_id}: Predicted next positions: {preds}")
