# Chapter 17: Backpropagation & Gradient Descent

> “*A neural network learns by looking backward—adjusting the past to improve the future.*”

---

This chapter reveals the mechanism that powers all deep learning: backpropagation.

We will:

- Understand how gradients are computed using the chain rule  
- Visualize how errors flow backward in a network  
- See how gradient descent uses those gradients to update weights  
- Use tf.GradientTape to track gradients  
- Step through a simplified manual backpropagation demo

## The Core Idea: Chain Rule

Backpropagation uses the chain rule from calculus to compute the gradient of the loss function with respect to each parameter in the network.

In a simple neural net:

```text
Input x → [W1, b1] → hidden → [W2, b2] → output → loss
```

We want to know:

```text
∂Loss/∂W1, ∂Loss/∂b1, ∂Loss/∂W2, ∂Loss/∂b2
```
TensorFlow does this using automatic differentiation.

---

##  tf.GradientTape: TensorFlow’s Engine

```python
with tf.GradientTape() as tape:
    logits = model(x_batch)
    loss = loss_fn(y_batch, logits)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```
- GradientTape watches all operations to record them.  
- When tape.gradient() is called, TensorFlow traces those operations backward using the chain rule.

---

## Manual Backpropagation: A Tiny Example

Let’s build a single-layer model manually and compute gradients ourselves.

1. Define Inputs and Parameters

```python
x = tf.constant([[1.0, 2.0]])
y_true = tf.constant([[1.0]])

W = tf.Variable([[0.1], [0.2]])
b = tf.Variable([0.3])
```

2. Forward Pass and Loss

```python
def forward(x):
    return tf.matmul(x, W) + b

def mse(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))
```

3. Compute Gradients

```python
with tf.GradientTape() as tape:
    y_pred = forward(x)
    loss = mse(y_pred, y_true)

grads = tape.gradient(loss, [W, b])
```

4. Apply Gradients

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer.apply_gradients(zip(grads, [W, b]))
```
This is **manual backpropagation** with a single-layer network. The same process scales up to thousands of layers internally!

---

## Gradient Descent: The Learning Step

At every training step, gradient descent does this:

```python
new_weight = old_weight - learning_rate * gradient
```
Variants like **Adam**, **RMSProp**, etc., optimize this update rule by adapting learning rates.

---

## Intuition: Why Does This Work?

Imagine trying to descend a mountain blindfolded, feeling the slope with your feet. Gradient descent gives you the direction (steepest descent) and a step size. Backpropagation tells you how each step affects your overall position (loss).

Together, they let the network learn even in high-dimensional, abstract spaces.

---

## Summary

In this chapter, we:  

- Demystified backpropagation using the chain rule  
- Used `tf.GradientTape` to compute gradients automatically  
- Performed a step-by-step manual backpropagation  
- Understood how gradient descent updates weights toward lower loss

Backpropagation isn’t just a technique—it’s the soul of deep learning. Mastering it gives you power to customize and debug any neural architecture.