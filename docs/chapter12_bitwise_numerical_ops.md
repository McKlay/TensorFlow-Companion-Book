---
hide:
  - toc
---

# Chapter 12: Bitwise & Numerical Operations

> “*Machine learning is built on numbers—but sharpened with operations.*”

---

## 12.1 Why These Ops Matter

These operations may seem low-level, but they power:  

- Loss functions  
- Data normalization  
- Activation functions  
- Efficient GPU/TPU processing  
- Feature engineering & logical masking  

If you want full control over your data pipeline or model internals, this is your toolkit.

---

## 12.2 Numerical Operations

TensorFlow supports a full range of math operations:

### ✅ Element-wise Math:
```python
x = tf.constant([1.0, 2.0, 3.0])

print(tf.math.square(x))     # [1, 4, 9]
print(tf.math.sqrt(x))       # [1.0, 1.4142, 1.7320]
print(tf.math.exp(x))        # Exponential
print(tf.math.log(x))        # Natural log
```
### ✅ Reduction Ops:
```python
matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(tf.reduce_sum(matrix))           # 10.0
print(tf.reduce_mean(matrix))          # 2.5
print(tf.reduce_max(matrix, axis=0))   # [3.0, 4.0]
```
> 💡 Reduction ops collapse tensors along a specified axis.

---

## 12.3 Rounding & Clipping

```python
a = tf.constant([1.2, 2.5, 3.8])

print(tf.round(a))       # [1.0, 2.0, 4.0]
print(tf.floor(a))       # [1.0, 2.0, 3.0]
print(tf.math.ceil(a))   # [2.0, 3.0, 4.0]
print(tf.clip_by_value(a, 1.5, 3.0))  # [1.5, 2.5, 3.0]
```

---

## 12.4 Bitwise Operations (for Integers Only)

Bitwise operations are useful for:  
- Masks and binary logic  
- Pixel manipulation (in image tasks)  
- Efficient boolean filters

```python
x = tf.constant([0b1010, 0b1100], dtype=tf.int32)
y = tf.constant([0b0101, 0b1010], dtype=tf.int32)

print(tf.bitwise.bitwise_and(x, y))  # [0b0000, 0b1000]
print(tf.bitwise.bitwise_or(x, y))   # [0b1111, 0b1110]
print(tf.bitwise.invert(x))          # Bitwise NOT
print(tf.bitwise.left_shift(x, 1))   # Shift left (×2)
print(tf.bitwise.right_shift(x, 1))  # Shift right (÷2)
```

---

## 12.5 Modulo and Sign

```python
print(tf.math.mod(17, 5))         # 2
print(tf.math.floormod(17, 5))    # 2
print(tf.math.sign([-2.0, 0.0, 3.0]))  # [-1.0, 0.0, 1.0]
```
These are useful in:  

- Cycle detection  
- Position encoding  
- Boolean masks for data pipelines

---

## 12.6 One-Hot Encoding (Bonus)

```python
labels = tf.constant([0, 2, 1])
one_hot = tf.one_hot(labels, depth=3)
print(one_hot)
```
### Output:
```lua
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]
```
This is crucial for classification tasks before training.

---

## 12.7 Summary

- TensorFlow supports a wide range of numerical, reduction, and bitwise operations.  
- These ops form the foundation for loss computation, feature preprocessing, and low-level tensor control.  
- Mastering them helps you go beyond layers—into the math powering them.

---

> “*Machine learning is built on numbers—but sharpened with operations.*”

---

## End of Part II: Tensor Mechanics and Computation

You now know how to:  

- Slice, reshape, and broadcast tensors  
- Work with ragged, sparse, and string data  
- Create trainable variables  
- Record and compute gradients  
- Write high-performance TensorFlow graphs