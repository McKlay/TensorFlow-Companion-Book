# Chapter 3: TensorFlow vs Keras

> “Abstraction isn’t just convenience—it’s control disguised as simplicity.”

---

## 3.1 The Relationship: Is Keras Part of TensorFlow?

Short answer: Yes. But not always.
Keras started as an independent deep learning API in 2015, aiming to make neural networks user-friendly. Originally, it supported multiple backends like TensorFlow, Theano, and CNTK.

But in 2017, Google integrated Keras as TensorFlow’s official high-level API, now accessed as:
```python
import tensorflow as tf
from tensorflow import keras
```
You’ll often see it written as tf.keras—that’s the TensorFlow-native version. And it’s not just a wrapper anymore—it’s part of TensorFlow’s core ecosystem.

---

## 3.2 Key Differences and Integration Points

Let’s compare them side by side:

|Feature	                    |tf.keras	                                    |Standalone Keras (pip install keras)   |
|-------------------------------|-----------------------------------------------|---------------------------------------|
|Backend Engine	                |TensorFlow only	                            |Can run on TensorFlow, Theano, etc.    |
|Performance Optimizations	    |Supports @tf.function, XLA, mixed-precision	|Limited (no deep TF graph integration) |
|Distributed Training	        |Integrated with tf.distribute.Strategy	        |Manual or not available                |
|Mobile Deployment	            |TFLite-compatible	                            |Not directly                           |
|Versioning	                    |Follows TensorFlow versioning	                |Versioned independently                |
|Recommended?	                |✅ Yes, always use tf.keras today	           |❌ Deprecated for most TF users       |

---

## 3.3 Why TensorFlow Adopted Keras

Because building models directly with low-level TensorFlow APIs was... painful.
```python
# Old-school TF 1.x style (yikes)
W = tf.Variable(tf.random.normal((784, 64)))
b = tf.Variable(tf.zeros(64))
y = tf.nn.relu(tf.matmul(x, W) + b)
```

Now with `tf.keras`:
```python
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,))
])
```
Cleaner. Safer. Reusable. And it connects directly to:  

- Loss functions (tf.keras.losses)  
- Optimizers (tf.keras.optimizers)  
- Metrics (tf.keras.metrics)  
- Model training (model.fit, model.evaluate, etc.)

---

## 3.4 The Layer Cake of TensorFlow

```lua
     [ tf.keras (user API) ]  
                ↓  
  [ Computation Graph (tf.function) ]  
                ↓  
    [ Core TensorFlow Engine ]  
                ↓  
  [ Hardware Execution (CPU/GPU/TPU) ]  
```

The beauty of `tf.keras` is this: you stay at the top layer, and TensorFlow handles the heavy machinery underneath.

You can still drop down to lower levels when needed—but `tf.keras` is designed for 90% of use cases.

---

## 3.5 When to Use tf.keras vs Raw TensorFlow

|Situation	                                |Use tf.keras?	        |Use raw tf?        |
|-------------------------------------------|-----------------------|-------------------|
|Building standard neural nets	            |✅ Yes	               |❌ Too verbose    |
|Custom ops / experimental graph behavior	|❌ Maybe	           |✅ Yes            |
|Writing production models	                |✅ Yes	               |✅ Sometimes      |
|Debugging gradients manually	            |✅ + GradientTape	   |✅                |

Use `tf.keras` unless you're experimenting with internals or need full control.

---

> “Keras makes TensorFlow human—so you can focus on ideas, not boilerplate.”