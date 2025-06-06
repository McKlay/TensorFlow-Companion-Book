---
hide:
  - toc
---

# Chapter 6: Tensor Indexing & Reshaping

“Tensors may be infinite in dimension, but mastery begins with the first slice.”

---

## 6.1 Why Indexing & Reshaping Matter

Before you train a single model, you’ll spend a good chunk of time doing this:

- Selecting rows, columns, or channels
- Flattening or expanding shapes
- Swapping axes
- Prepping tensors for layers like Dense, Conv2D, or RNN

Think of this as data martial arts—getting your tensors into the right stance before the real fight begins.

---

## 6.2 Basic Indexing (Rank 1 and 2)

```python
import tensorflow as tf

# Rank 1 (vector)
vec = tf.constant([10, 20, 30, 40])
print(vec[0])     # 10
print(vec[-1])    # 40

# Rank 2 (matrix)
mat = tf.constant([[1, 2], [3, 4], [5, 6]])
print(mat[1])        # [3, 4]
print(mat[1, 0])     # 3
```
Slicing also works:
```python
print(mat[:, 0])     # First column: [1, 3, 5]
print(mat[0:2, :])   # First two rows: [[1, 2], [3, 4]]
```

---

## 6.3 Reshaping Tensors

`tf.reshape()` lets you change a tensor’s shape without changing its data:
```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
reshaped = tf.reshape(x, [3, 2])         # Shape: (3, 2)
print(reshaped)
```
You can also flatten:
```python
flat = tf.reshape(x, [-1])  # Automatically infer length
print(flat)  # [1, 2, 3, 4, 5, 6]
```

---

## 6.4 Expanding and Squeezing Dimensions

These are crucial when batching data or feeding into specific layer shapes:
```python
x = tf.constant([1, 2, 3])  # Shape: (3,)

# Expand
x_expanded = tf.expand_dims(x, axis=0)  # Shape: (1, 3)
x_expanded2 = tf.expand_dims(x, axis=1) # Shape: (3, 1)

# Squeeze
x_squeezed = tf.squeeze(tf.constant([[1], [2], [3]]))  # Shape: (3,)
```
Use cases:

- expand_dims → simulate a batch: [3] → [1, 3]

- squeeze → remove unnecessary dimensions (e.g. from model outputs)

---

## 6.5 Transposing and Permuting Axes

You can swap dimensions using `tf.transpose()`:
```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
print(tf.transpose(x))  # Shape: (3, 2)
```
For higher-rank tensors, use `perm`:
```python
x = tf.random.normal([2, 3, 4])
x_transposed = tf.transpose(x, perm=[0, 2, 1])  # Swaps last two dims
```

---

## 6.6 Tensor Shape Tricks You’ll Actually Use

|Goal	                        |Command                                |
|-------------------------------|---------------------------------------|
|Flatten a tensor	            |tf.reshape(tensor, [-1])               |
|Add batch dimension	        |tf.expand_dims(tensor, axis=0)         |
|Remove singleton dims	        |tf.squeeze(tensor)                     |
|Change channel-last to first	|tf.transpose(tensor, [0, 3, 1, 2])     |
|Recover original shape	        |tf.reshape(tensor, orig_shape)         |

---

## 6.7 Summary

- Indexing lets you extract elements, rows, columns, or slices from tensors of any rank.
- tf.reshape() allows you to safely change tensor shapes—crucial before feeding into models.
- expand_dims() and squeeze() help manage batch dimensions and singleton axes.
- transpose() and perm are useful for rearranging axes, especially in image and sequence data.
- Shape manipulation is not just a utility—it’s how you adapt data to flow through deep learning systems.

---

> “Tensors may be infinite in dimension, but mastery begins with the first slice.”


