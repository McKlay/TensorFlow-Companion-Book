# Chapter 26: Convolution Layers & CNNs

> “*If fully connected layers taught our models to think, convolutional layers taught them to see.*”

---

## Introduction: Why Convolution?

Imagine trying to recognize a cat in a picture by scanning every pixel individually. That’s what early neural networks did—and they weren’t great at it. But in 1998, LeNet-5 by Yann LeCun changed everything by introducing convolutional layers—special layers that act like visual pattern detectors.

Today, Convolutional Neural Networks (CNNs) are the backbone of modern computer vision. From facial recognition to autonomous vehicles, CNNs empower machines to see.

---

## Core Idea Behind Convolution:

Unlike fully connected layers where each neuron is connected to all inputs, convolution layers use small filters (kernels) to scan over input data. Each filter detects specific features like:

- Vertical edges  
- Horizontal lines  
- Patterns or textures

This drastically reduces the number of parameters and preserves spatial relationships—crucial for images.

---

## Implementing Convolution in TensorFlow

Here's how to use convolutional layers with tf.keras.layers.Conv2D:

---

### ✅ Code Example: Basic Conv2D Layer
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
**Explanation**

- `Conv2D`: 2D convolution layer with 32 filters of size 3x3  
- `activation='relu'`: adds non-linearity   
- `input_shape=(28,28,1)`: input is a grayscale 28x28 image  
- `MaxPooling2D`: downscales the feature map by taking the max value in a 2x2 window  
- `Flatten` and `Dense`: used for final classification

---

## Anatomy of a Conv Layer

|Component	            |Description                                                |
|-----------------------|-----------------------------------------------------------|
|Filters	            |Small matrices (e.g., 3x3) that slide over the image       |
|Stride	                |Steps taken by the filter as it moves                      |
|Padding	            |Adding borders to maintain output size (e.g., `'same'`)    |
|Activation	            |Usually ReLU to add non-linearity                          |
|Output Tensor	        |A 3D tensor: (`height`, `width`, `channels`)               |

---

##  Math Behind Convolution

Given:

- Input size: 28x28  
- Filter size: 3x3  
- Stride: 1  
- Padding: 'valid'  
- Then output size is:

```lua
    Output=⌊ (28−3+1)/1 ⌋ = 26×26
```

Each filter generates one feature map. With `filters=32`, you’ll get 32 feature maps stacked as the output tensor.

---

##  Stack of Convolutions: A Feature Hierarchy

As layers stack up:

- First layers detect edges  
- Middle layers detect patterns (like eyes or wheels)  
- Later layers detect objects (faces, dogs, cars)

---

## Try It Live: Google Colab

▶️ [Open in Colab](https://colab.research.google.com/)

---

## Summary

In this chapter, you learned how convolution layers work as the eyes of your model. You saw how:

- CNNs drastically reduce parameter count compared to dense networks  
- Conv2D layers extract features through filters  
- Feature maps evolve from edges to full object representations

---

