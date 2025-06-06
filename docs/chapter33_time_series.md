---
hide:
  - toc
---

# Chapter 33: Time Series Forecasting

> “*The best way to predict the future is to model the past — and time series helps us do just that.*”

---

## 🧭 Introduction: When Time Becomes a Feature

Time series data is everywhere — stock prices, weather data, electricity usage, web traffic, heart rate signals, you name it. What makes time series special is that order matters. Unlike other datasets, a time series isn’t just a collection of independent observations — it’s a sequence, where each point is dependent on the ones before it.

This chapter introduces how to handle time series forecasting using TensorFlow. We’ll move from simple concepts like trend and seasonality, all the way to training neural networks (including LSTM and CNN variants) to make future predictions.

---

## Key Concepts in Time Series

Before we jump into TensorFlow code, let’s quickly review the building blocks of time series:

- Trend: Long-term increase or decrease in the data  
- Seasonality: Repeating patterns over a fixed period (e.g., daily, monthly)  
- Noise: Randomness that we want to filter out  
- Lag: How past values affect future values (e.g., y(t) = y(t-1) + ε)

## Data Preparation

Here’s how to prepare a time series dataset for training:

### Step 1: Load the Data
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
df.columns = ['Month', 'Passengers']
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

df.plot()
plt.title("Monthly Airline Passengers")
plt.show()
```

### Step 2: Normalize & Window the Data

TensorFlow expects a supervised format — so we **convert time series into windows of inputs and labels**.
```python
import numpy as np

def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

series = df['Passengers'].values.astype(np.float32)
series = (series - series.mean()) / series.std()  # normalize
window_size = 12

X, y = create_dataset(series, window_size)
X = X[..., np.newaxis]  # [batch, time, features]
```

---

## Model 1: Dense Neural Network

Let’s start with a simple fully-connected network.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = tf.keras.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=1)
```

---

## Model 2: LSTM for Time Awareness

Recurrent models like LSTM are specifically designed for sequences.
```python
model = tf.keras.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.LSTM(64),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100)
```

---

## 📈 Visualizing the Prediction

After training, you can compare predictions to the actual values.
```python
pred = model.predict(X)

plt.plot(series[window_size:], label='Actual')
plt.plot(pred.squeeze(), label='Predicted')
plt.legend()
plt.title("Forecast vs Actual")
plt.show()
```

---

## Alternative Models You’ll Explore Later

- **1D CNNs**: For fast inference on long sequences  
- **Transformer Time Series Models**: For long-term memory  
- **Autoregressive RNNs**: Predict step-by-step into the future  
- **Hybrid Models**: Combine statistical models (ARIMA) with DL

---

## Summary

In this chapter, you learned:

- How time series forecasting differs from other prediction tasks  
- How to preprocess and window time series data  
- How to train Dense and LSTM models using TensorFlow  
- How to evaluate and visualize your predictions

Time series forecasting is one of the most powerful ML tools for decision-making in business, finance, and operations. As we build on this, you’ll be ready to tackle real datasets like Bitcoin prices, sales prediction, or even traffic flow — with TensorFlow as your lens into the future.

---

