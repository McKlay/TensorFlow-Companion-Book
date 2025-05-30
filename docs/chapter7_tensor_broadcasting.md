# Chapter 7: Tensor Broadcasting

> “Broadcasting is TensorFlow’s way of saying: ‘Relax, I’ve got this shape mismatch.’”

---

## 7.1 What is Broadcasting?

Broadcasting allows tensors with different shapes to participate in operations as if they had the same shape.  
It's like auto-expanding dimensions on the fly so that element-wise operations just work—without explicitly reshaping anything.  
It’s a core part of NumPy, PyTorch, and yes—TensorFlow.

---

## 7.2 Broadcasting in Action
Let’s take a practical example:
```python
import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])       # Shape: (2, 2)
b = tf.constant([10, 20])               # Shape: (2,)

result = a + b
print(result)
```
TensorFlow “broadcasts” b from shape (2,) → (2, 2) by duplicating it across rows:
```lua
[[1 + 10, 2 + 20],
 [3 + 10, 4 + 20]]
= [[11, 22],
   [13, 24]]
```

---

## 7.3 Broadcasting Rules (The Intuition)
TensorFlow compares dimensions from right to left:

If same, they’re compatible.

If one is 1, it’s stretched to match.

If they don’t match and neither is 1, it’s an error.  
**Example:**
```python
a: (4, 1, 3)
b: (  , 5, 1)
```
→ Resulting shape: `(4, 5, 3)`
The middle `1` in `a` stretches to `5`
The last `1` in `b` stretches to `3`

---

## 7.4 Common Broadcasting Use Cases

✅ Adding a bias vector to each row:
```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])
bias = tf.constant([10, 20, 30])
print(x + bias)
```
✅ Multiplying by a scalar:
```python
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(x * 2.5)
```
✅ Normalizing each feature (column-wise):
```python
x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
mean = tf.reduce_mean(x, axis=0)  # Shape: (2,)
print(x - mean)  # Subtracts mean from each row
```

---

## 7.5 When Broadcasting Fails

Some operations won’t broadcast if shapes are completely incompatible:
```python
a = tf.constant([1, 2, 3])      # Shape: (3,)
b = tf.constant([[1, 2], [3, 4]])  # Shape: (2, 2)

# tf.add(a, b) → ❌ Error: Shapes can't broadcast
```
Always check shape compatibility first:
```python
print("Shape A:", a.shape)
print("Shape B:", b.shape)
```
If needed, use:

- `tf.expand_dims()`

- `tf.reshape()`

- or `tf.broadcast_to()` to explicitly adjust shapes

---

## 7.6 Summary

- Broadcasting lets tensors with different shapes interact during operations.
- TensorFlow automatically stretches dimensions when one of them is 1.
- Most common use cases involve scalars, bias addition, and feature-wise normalization.
- Broadcasting removes boilerplate reshaping—just mind your axes.

---

> “Broadcasting is TensorFlow’s way of saying: ‘Relax, I’ve got this shape mismatch.’”