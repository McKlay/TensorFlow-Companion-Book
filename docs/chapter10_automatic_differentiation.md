---
hide:
  - toc
---

# Chapter 10: Automatic Differentiation `(tf.GradientTape)`

> “*The power to learn lies not in the function—but in how it changes*.”

---

## 10.1 What Is Automatic Differentiation?

Automatic differentiation (autodiff) is the core engine of backpropagation. It allows TensorFlow to:

- Record operations as a computation graph  
- Compute gradients w.r.t. any variable  
- Use those gradients to optimize parameters  

In TensorFlow, this is done using `tf.GradientTape`.

---

## 10.2 Recording with `tf.GradientTape`

```python
import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1  # y = (x + 1)^2

dy_dx = tape.gradient(y, x)
print("dy/dx:", dy_dx.numpy())  # Should be 2x + 2 → 8.0
```
> By default, tape only watches `tf.Variables`.

---

## 10.3 Watching Non-Variable Tensors

If you want to differentiate w.r.t. a tensor (not a variable):
```python
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2

dy_dx = tape.gradient(y, x)
print("dy/dx:", dy_dx.numpy())  # 6.0
```

---

## 10.4 Multivariate Gradients

```python
x = tf.Variable(1.0)
y = tf.Variable(2.0)

with tf.GradientTape() as tape:
    f = x**2 + y**3

grads = tape.gradient(f, [x, y])
print("df/dx:", grads[0].numpy())  # 2x → 2
print("df/dy:", grads[1].numpy())  # 3y^2 → 12
```
You can compute gradients for multiple variables at once—essential in model training.

---

## 10.5 Persistent Tapes (Reuse for Multiple Gradients)

```python
x = tf.Variable(1.0)

with tf.GradientTape(persistent=True) as tape:
    y = x**2
    z = x**3

dy_dx = tape.gradient(y, x)  # 2x
dz_dx = tape.gradient(z, x)  # 3x^2
print(dy_dx.numpy(), dz_dx.numpy())  # 2.0, 3.0

del tape  # Cleanup
```
Use `persistent=True` if you need to compute multiple gradients from the same tape.

---

## 10.6 Using Gradients for Optimization

```python
x = tf.Variable(2.0)

learning_rate = 0.1
for i in range(10):
    with tf.GradientTape() as tape:
        loss = (x - 5)**2

    grad = tape.gradient(loss, x)
    x.assign_sub(learning_rate * grad)

    print(f"Step {i}, x: {x.numpy():.4f}, loss: {loss.numpy():.4f}")
```
This mimics gradient descent—updating x to minimize the loss.

---

## 10.7 Summary

- tf.GradientTape is TensorFlow’s way of recording operations and computing gradients automatically.
- It’s the core of learning—used in every optimizer.
- Gradients are computed via tape.gradient(loss, variables)
- You can track multiple variables, persist the tape, or manually apply updates.

---

> “*The power to learn lies not in the function—but in how it changes.*”
