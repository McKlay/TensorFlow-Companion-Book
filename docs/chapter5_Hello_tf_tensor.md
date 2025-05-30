# Chapter 5: Hello, tf.Tensor!

“All of machine learning boils down to manipulating tensors—smartly.”

---

## 5.1 What is a Tensor?
Think of a tensor as a container for numbers, with a specific number of dimensions (aka rank).

- Scalar → Rank 0 → Just a number
- Vector → Rank 1 → A list of numbers
- Matrix → Rank 2 → Rows and columns
- Tensor → Rank ≥3 → Multi-dimensional data (e.g. images, videos, batches)
- Everything in TensorFlow revolves around these.

---

## 5.2 Creating Tensors
Let’s create some:
```python
import tensorflow as tf

# Scalar (Rank 0)
scalar = tf.constant(42)

# Vector (Rank 1)
vector = tf.constant([1.0, 2.0, 3.0])

# Matrix (Rank 2)
matrix = tf.constant([[1, 2], [3, 4]])

# 3D Tensor
tensor3d = tf.constant([[[1], [2]], [[3], [4]]])
```
Use `tf.constant()` to create immutable tensors.
Use `tf.Variable()` if you want a trainable version.

---

## 5.3 Exploring Tensors

Let’s peek inside:
```python
print("Shape:", tensor3d.shape)
print("Rank (ndim):", tf.rank(tensor3d))
print("DType:", tensor3d.dtype)
print("Device:", tensor3d.device)
```
Output might look like:
```vbnet
Shape: (2, 2, 1)
Rank (ndim): tf.Tensor(3, shape=(), dtype=int32)
DType: <dtype: 'int32'>
```

---

## 5.4 Type and Shape Manipulation

TensorFlow is strict about data types and shapes—get comfortable doing this:
```python
# Cast to float32
float_tensor = tf.cast(matrix, dtype=tf.float32)

# Reshape (e.g. flatten 2x2 → 4)
reshaped = tf.reshape(matrix, [4])

# Expand dimensions (useful for batch simulation)
expanded = tf.expand_dims(vector, axis=0)  # Now shape = (1, 3)

# Squeeze dimensions
squeezed = tf.squeeze(tensor3d)
```

---

## 5.5 Basic Tensor Operations

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Element-wise
print(tf.add(a, b))        # [5 7 9]
print(tf.multiply(a, b))   # [4 10 18]

# Matrix multiplication
mat1 = tf.constant([[1, 2], [3, 4]])
mat2 = tf.constant([[5, 6], [7, 8]])
print(tf.matmul(mat1, mat2))  # [[19 22], [43 50]]
```
> 💡 TensorFlow broadcasts shapes automatically if they align—more on that in Chapter 7.

---

## 5.6 Inspecting and Debugging

For development, you’ll often want to inspect:
```python
print(tf.shape(matrix))     # Tensor with shape info
print(matrix.numpy())       # Convert to NumPy for debugging
```

You can convert any tensor to a NumPy array using .numpy()—especially useful when running in eager mode.

---

> “All of machine learning boils down to manipulating tensors—smartly.”
