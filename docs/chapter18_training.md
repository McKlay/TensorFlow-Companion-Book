---
hide:
  - toc
---

# Chapter 18: Model Training with fit() and evaluate()

> “*With great abstraction comes great productivity. `fit()` lets you train deep models with a single line.*”

---

After understanding the fundamentals of backpropagation and gradient descent, you now know what goes on under the hood. In this chapter, we shift our focus to using TensorFlow’s high-level training interface—the `fit()` and `evaluate()` methods from the `tf.keras.Model` class.

By the end, you’ll be able to:

- Train models with `model.fit()` using real datasets  
- Track performance with `model.evaluate()` and `model.predict()`  
- Customize training with validation splits, callbacks, and batch sizes  
- Monitor overfitting and training speed

---

## Setup: A Simple Model

Let’s use the **MNIST** dataset and a fully connected neural net.

```python
import tensorflow as tf

# Load & preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
```

---

## Training with fit()

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
```
- epochs: number of full passes through the training data  
- batch_size: how many samples per gradient update  
- validation_split: % of training data used for validation

---

## Evaluation and Prediction

Once trained, evaluate the model:

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
```

Make predictions:

```python
logits = model.predict(x_test)
predictions = tf.argmax(logits, axis=1)
```

---

## Using Callbacks

Callbacks allow you to hook into the training process. Common uses:

- Early stopping  
- Model checkpointing  
- Logging

```python
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(
    x_train, y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[callback]
)
```

---

## Custom Batching with `tf.data`

For more control, use `tf.data.Dataset`:
```python
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1024).batch(64)

model.fit(train_ds, epochs=5)
```
This is especially useful for large datasets, streaming data, or augmentations.

---

## Monitoring Training

To track progress visually, use the `history` object:
```python
history = model.fit(...)
```
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
```

---

## Summary

In this chapter, you learned:

- How `fit()` abstracts the entire training loop  
- How to validate, evaluate, and predict using simple APIs  
- How callbacks improve training flexibility  
- How to use `tf.data` for custom batching

The `fit()` interface doesn’t replace your understanding of backprop—it supercharges it. Now you can focus on what to train, not just how to train it.

---
