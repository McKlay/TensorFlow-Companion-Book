# Chapter 16: Loss Functions & Optimizers

> “*Without a compass, even the smartest network gets lost. Loss guides learning. Optimization moves us forward.*”

---

In this chapter, we explore two of the most crucial ingredients of any machine learning recipe:

- Loss functions: Measure how far off our model’s predictions are from the actual values.  
- Optimizers: Algorithms that adjust model parameters to minimize this loss.

By the end, you'll understand:

- The difference between various loss functions and when to use them  
- How gradients are computed and used  
- Popular optimization algorithms and their trade-offs  
- How to implement custom loss functions and plug them into training

---

## What Is a Loss Function?

A loss function tells us how “bad” our predictions are. It is scalar-valued, allowing TensorFlow to compute gradients via backpropagation.

🔹 Common Losses in TensorFlow

|Task	                        |Loss Function	                    |TensorFlow API                                     |
|-------------------------------|-----------------------------------|---------------------------------------------------|
|Binary classification	        |Binary Crossentropy	            |tf.keras.losses.BinaryCrossentropy()               |
|Multi-class classification	    |Sparse Categorical Crossentropy	|tf.keras.losses.SparseCategoricalCrossentropy()    |
|Regression (real values)	    |Mean Squared Error	                |tf.keras.losses.MeanSquaredError()                 |

#### Example: Sparse Categorical Crossentropy

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_true, y_pred)
```
- `from_logits=True` means the model outputs raw values (logits) without softmax.

- If your model outputs softmax-activated values, set f`rom_logits=False`.

---

## What Are Optimizers?

Optimizers update model parameters using gradients computed from the loss. They are essential for gradient descent-based training.

🔹 Popular Optimizers

|Optimizer	    |Description	                                            |Usage                          |
|---------------|-----------------------------------------------------------|-------------------------------|
|SGD	        |Stochastic Gradient Descent	                            |SGD(learning_rate=0.01)        |
|Momentum	    |Adds inertia to SGD	                                    |SGD(momentum=0.9)              |
|RMSProp	    |Adjusts learning rate based on recent magnitudes	        |RMSprop(learning_rate=0.001)   |
|Adam	        |Combines Momentum + RMSProp	                            |Adam(learning_rate=0.001)      |

#### Example: Compile with Optimizer

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

---

## Custom Loss Function

Sometimes, built-in loss functions aren’t enough. Here’s how you can define your own:

```python
def custom_mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
```

Plug into model like this:

```python
model.compile(
    optimizer='adam',
    loss=custom_mse_loss
)
```

---

## Custom Training Loop (Optional Recap)

When not using model.fit(), you need to compute loss and apply gradients manually:

```python
with tf.GradientTape() as tape:
    logits = model(x_batch, training=True)
    loss_value = loss_fn(y_batch, logits)

grads = tape.gradient(loss_value, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```
This gives you full control over training and is often used in research or advanced custom workflows.

---

## Summary

In this chapter, you learned:

- Loss functions quantify how wrong a model’s predictions are.  
- Optimizers use gradients to update model weights and minimize loss.  
- Adam is a great default optimizer, but others may work better depending on the problem.  
- You can define custom loss functions for flexibility.  

Understanding the relationship between loss → gradient → optimizer → new weights is the key to mastering how neural networks learn.

---
