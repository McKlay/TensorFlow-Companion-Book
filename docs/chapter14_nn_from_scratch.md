# Chapter 14: Building a Neural Network from Scratch

> “*Before you rely on magic, understand the machinery beneath it.*”

---

In this chapter, we'll strip away the abstraction of high-level APIs and dive into the inner mechanics of building a neural network step-by-step using only low-level TensorFlow operations (tf.Variable, tf.matmul, tf.nn, etc.). This exercise gives you a deeper appreciation of what libraries like tf.keras automate for us—and how neural networks actually operate under the hood.

By the end of this chapter, you’ll be able to:

- Initialize weights and biases manually  
- Write your own forward pass function  
- Calculate loss and accuracy  
- Implement backpropagation using tf.GradientTape  
- Train a minimal network on a real dataset (e.g., MNIST)  

---

## Step 1: Dataset Preparation

We’ll use the MNIST dataset (handwritten digits) for simplicity. It's preloaded in TensorFlow:
```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert to tf.Tensor
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
```

---

## Step 2: Model Initialization

We'll define a simple feedforward neural network with:

- Input layer: 784 units (28x28 pixels)  
- Hidden layer: 128 units + ReLU  
- Output layer: 10 units (one per digit)

```python
# Parameters
input_size = 784
hidden_size = 128
output_size = 10

# Weights and biases
W1 = tf.Variable(tf.random.normal([input_size, hidden_size], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_size]))
W2 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
b2 = tf.Variable(tf.zeros([output_size]))
```

---

##  Step 3: Forward Pass Function

```python
def forward_pass(x):
    hidden = tf.nn.relu(tf.matmul(x, W1) + b1)
    logits = tf.matmul(hidden, W2) + b2
    return logits
```

---

##  Step 4: Loss & Accuracy

Use sparse categorical cross-entropy since labels are integer-encoded:
```python
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
```

---

## Step 5: Training Loop

Now we manually implement the training loop using `tf.GradientTape`.

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 5
batch_size = 64

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            logits = forward_pass(x_batch)
            loss = compute_loss(logits, y_batch)

        gradients = tape.gradient(loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))

    # Epoch-end evaluation
    test_logits = forward_pass(x_test)
    test_acc = compute_accuracy(test_logits, y_test)
    print(f"Epoch {epoch+1}, Test Accuracy: {test_acc:.4f}")
```

---

## Summary

In this chapter, we:  

- Built a fully functioning neural network without tf.keras  

- Initialized all parameters manually  

- Defined forward propagation, loss, and backpropagation  

- Trained it on MNIST using gradient descent  

Understanding how to manually construct and train a neural network builds foundational intuition that will help you:

- Debug custom layers and losses  

- Understand performance bottlenecks  

- Transition into low-level model tweaking when needed

---
