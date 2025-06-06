---
hide:
  - toc
---

# Chapter 15: Layers & Activation Functions

> “*Neurons speak in activations. Layers translate

---

High-level APIs in TensorFlow make it easy to build neural networks using predefined layers and activation functions. But beneath these abstractions lies the mathematical logic we explored in Chapter 14.

This chapter dives into:

- Commonly used layers in tf.keras.layers  
- How layers are composed in Sequential and Functional APIs  
- The role and behavior of activation functions  
- How to visualize activation functions  
- Choosing the right activation based on task  

By the end, you’ll grasp how layers and activations form the “building blocks” of deep learning architectures.

---

## 🏗️ Understanding Layers

Layers are wrappers around mathematical functions that transform input tensors. Common types include:

#### 1. Dense Layer
Fully-connected layer where every neuron receives input from all neurons in the previous layer.

```python
from tensorflow.keras.layers import Dense

dense_layer = Dense(units=64, activation='relu')
```

#### 2. Dropout Layer
Randomly disables a fraction of units during training to prevent overfitting.

```python
from tensorflow.keras.layers import Dropout

dropout_layer = Dropout(rate=0.5)  # Drops 50% of units
```

#### 3. Flatten Layer
Converts multi-dimensional input (e.g., 28x28 image) to 1D vector.

```python
from tensorflow.keras.layers import Flatten

flatten_layer = Flatten()
```

---

## ⚡ Activation Functions

Activation functions determine whether a neuron “fires” or not. They introduce non-linearity—critical for learning complex patterns.

#### 🔹 Common Activation Functions
|Function	        |Formula	                           |Use Case                                    |
|-------------------|--------------------------------------|--------------------------------------------|
|ReLU	            |max(0, x)	                           |Most common; avoids vanishing gradients     |
|Sigmoid	        |1 / (1 + e^-x)	                       |Binary classification (last layer)          |
|Tanh	            |(e^x - e^-x) / (e^x + e^-x)	       |Good zero-centered activation               |
|Softmax	        |exp(x_i) / Σ exp(x_j) (multi-class)   |Output layer for multi-class problems       |
|Leaky ReLU	        |x if x > 0 else α * x (α ~ 0.01)	   |Avoids dead ReLU units                      |


## Visualizing Activations

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x = tf.linspace(-10.0, 10.0, 400)

activations = {
    "ReLU": tf.nn.relu(x),
    "Sigmoid": tf.nn.sigmoid(x),
    "Tanh": tf.nn.tanh(x),
    "Leaky ReLU": tf.nn.leaky_relu(x),
}

plt.figure(figsize=(10, 6))
for name, y in activations.items():
    plt.plot(x, y, label=name)
plt.legend()
plt.title("Activation Functions")
plt.grid(True)
plt.show()
```

---

## Build a Network with Layers & Activations

Here’s how you can simplify your neural net from Chapter 14 using layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)  # No softmax here; included in loss
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()
```

---

## Functional API Example

The Functional API allows for more complex architectures (e.g., multi-input, multi-output):

```python
from tensorflow.keras import Model, Input

inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)
outputs = Dense(10)(x)

model = Model(inputs=inputs, outputs=outputs)
```

---

## Summary

In this chapter, you:

- Explored key layer types (Dense, Flatten, Dropout)  
- Learned how activation functions shape neural computations  
- Built models using Sequential and Functional APIs  
- Visualized and compared activation behaviors  

You now understand how layers and activations turn your raw tensors into meaningful representations that a neural network can learn from.

---