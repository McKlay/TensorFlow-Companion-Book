# Chapter 9: Variables & Trainable Parameters

> “*A model learns by changing its variables—not its mind*.”

---

## 9.1 What is a `tf.Variable`?  

While tf.Tensor is immutable, a `tf.Variable` is mutable—its values can be updated during training.

Think:
- Weights  
- Biases  
- Embeddings

All of these are backed by `tf.Variable`.

---

## 9.2 Creating Variables

✅ Basic Example:
```python
import tensorflow as tf

# Create a scalar variable
w = tf.Variable(3.0)

# Vector variable
v = tf.Variable([1.0, 2.0, 3.0])
```
You can inspect:
```python
print("Initial value:", v.numpy())
print("Shape:", v.shape)
print("Trainable:", v.trainable)
```

---

## 9.3 Updating Variables

You can assign new values or add/subtract in-place:
```python
w.assign(5.0)
w.assign_add(1.0)
w.assign_sub(2.0)
print(w.numpy())  # Output: 4.0
```

This is what optimizers do under the hood:
Update weights using gradients, via `assign_sub()`.

---

## 9.4 tf.Variable vs tf.Tensor

|Feature	                |`tf.Tensor`	      |`tf.Variable`              |
|---------------------------|-----------------|                         |
|Immutable	                |✅ Yes	        |❌ No                    |
|Used for constants	        |✅ Yes	        |🚫 Not recommended       |
|Learns in training	        |❌ No	        |✅ Yes                   |
|Used for model weights	    |❌	            |✅ Yes                   |

Use `tf.Variable` when you want TensorFlow to track the state during training.

---

## 9.5 Variables in Custom Training Loops

```python
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    loss = x**2 + y**2

grads = tape.gradient(loss, [x, y])
print("Gradients:", grads)

# Manually update
x.assign_sub(0.1 * grads[0])
y.assign_sub(0.1 * grads[1])
```
This mimics a single gradient descent step!

---

## 9.6 Variable Collections (Bonus)

TensorFlow tracks variables using:

- `tf.trainable_variables()`
- `model.trainable_variables` (in Keras models)

This helps optimizers know what to update during `.fit()` or `apply_gradients`.

---

## 9.7 Summary

- `tf.Variable` stores trainable state—used for weights, biases, and embeddings.  
- You can mutate them via `assign`, `assign_add`, and `assign_sub`.  
- `tf.Tensor` is for static data, `tf.Variable` is for learning state.  
- They're essential for building models, training loops, and optimization.

---

> “*A model learns by changing its variables—not its mind*.”
