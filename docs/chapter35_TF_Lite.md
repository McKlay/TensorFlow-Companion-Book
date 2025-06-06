---
hide:
  - toc
---

# Chapter 35: TensorFlow Lite & Mobile Deployment

> “*A model isn’t truly intelligent until it fits in your pocket.*”

---

## 📱 Introduction: Why Deploy on Mobile?

TensorFlow Lite (TFLite) is TensorFlow’s lightweight version tailored for **mobile and edge devices** like Android, iOS, Raspberry Pi, and microcontrollers. With the rise of real-time AI on phones (face unlock, voice assistants, image classifiers), deploying efficient models to run locally is now a core part of ML engineering.

---

In this chapter, you’ll learn how to:

- Convert TensorFlow models to `.tflite`  
- Optimize for performance with quantization  
- Deploy and test on Android/iOS  
- Run inference in Python or embedded systems

---

## Step 1: Convert a TensorFlow Model to TFLite

We’ll begin with a simple image classifier (e.g., trained on MNIST or MobileNet).
```python
import tensorflow as tf

# Load or train a model (example: MobileNetV2 pretrained)
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("mobilenetv2.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## Step 2: Optimize the Model with Quantization

Reducing model size and improving speed, especially for devices with limited resources.
```python
# Dynamic Range Quantization (fastest + simplest)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open("mobilenetv2_quant.tflite", "wb") as f:
    f.write(quantized_model)
```
Other options:

- Post-training integer quantization

- Full integer quantization (with representative dataset)

- Float16 quantization for GPUs

---

## Step 3: Run Inference with TFLite Interpreter (Python)

```python
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_quant.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare image input
img = Image.open("cat.jpg").resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Prediction:", np.argmax(output_data))
```

---

## 📲 Step 4: Deploy to Android/iOS

You can now take your `.tflite` model and integrate it into:

- **Android Studio** using `TensorFlow Lite Task Library` or `Interpreter API`

- **iOS apps** using Swift or Objective-C

- **Flutter apps** using the `tflite_flutter` plugin

💡 Bonus: Use ML Model Binding in Android Studio for no-code integration of `.tflite` models with image or text inputs!

---

## Other Deployment Targets

TensorFlow Lite isn’t just for phones:

|Platform	        |Use Case                                           |
|-------------------|---------------------------------------------------|
|Raspberry Pi	    |Smart cameras, IoT vision                          |
|Coral Edge TPU	    |Ultra-fast inference with acceleration             |
|Microcontrollers	|TinyML with TensorFlow Lite for Microcontrollers   |

---

## Summary

In this chapter, you learned:

- How to convert Keras models into efficient .tflite files

- How to optimize models with quantization

- How to run inference in Python, Android, or iOS

- That TFLite opens the door to AI at the edge — fast, offline, and private

TensorFlow Lite brings ML to environments where connectivity and power are limited, but the need for real-time intelligence is high — from smartwatches and cameras to drones and industrial machines.

---

