---
hide:
  - toc
---

# Chapter 20: Visualizing Model Progress with TensorBoard

> “*What you can’t see, you can’t debug. TensorBoard turns numbers into insight.*”

---

Training deep learning models involves a flurry of metrics, graphs, losses, weights, and gradients. TensorBoard is TensorFlow’s built-in visualization toolkit—your window into the learning process.

In this chapter, you’ll:

- Learn how to launch TensorBoard  
- Track training metrics, histograms, and model graphs  
- Visualize embeddings and profiler data  
- Log custom scalars, images, and text during training  
- Debug and tune your models more efficiently

## What Is TensorBoard?

TensorBoard is a web-based dashboard that visualizes:

- Loss and accuracy curves  
- Weight and bias distributions  
- Learning rates and gradients  
- Computation graphs  
- Embeddings in 2D/3D  
- Profiling info (CPU/GPU usage)

It helps:  
✅ Debug your training  
✅ Monitor model convergence  
✅ Tune hyperparameters  
✅ Share results with others

---

## Step 1: Setup a Logging Directory

```python
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```

Attach it to `fit()`:
```python
model.fit(
    x_train, y_train,
    epochs=5,
    validation_split=0.2,
    callbacks=[tensorboard_cb]
)
```

---

## Step 2: Launch TensorBoard

In terminal:
```bash
tensorboard --logdir logs/fit
```

Or in notebooks:
```bash
%load_ext tensorboard
%tensorboard --logdir logs/fit
```
Visit: `http://localhost:6006`
(Or open the hosted link in Colab)

---

## What Can You Visualize?

#### 🔹 Scalars
Loss, accuracy, learning rate, etc.

#### 🔹 Histograms
Distributions of weights, biases, activations.

#### 🔹 Graph
Model computation graph for debugging layers and shapes.

#### 🔹 Embeddings
Visualize high-dimensional features (like word embeddings) using t-SNE or PCA.

#### 🔹 Images
Track model outputs visually, e.g., generated images.

```python
file_writer = tf.summary.create_file_writer(log_dir + "/images")

with file_writer.as_default():
    tf.summary.image("Sample images", sample_batch, step=0)
```

---

## Profile Model Performance

You can enable profiling to analyze slowdowns:
```python
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='500,520'  # Profile between these batches
)
```
This provides:

- CPU/GPU usage  
- Memory footprint  
- Bottlenecks in data loading or model ops  

---

## Custom Logging (Advanced)

Log custom scalars or text:
```python
with file_writer.as_default():
    tf.summary.scalar('custom_metric', value, step=epoch)
    tf.summary.text('note', 'Validation dropped sharply!', step=epoch)
```

---

## Summary

In this chapter, you:

- Set up TensorBoard to monitor training  
- Explored scalar, histogram, and graph visualizations  
- Learned to profile and debug performance  
- Logged custom metrics for advanced diagnostics

TensorBoard transforms your training sessions into stories, making every epoch readable, trackable, and shareable. It’s a must-have for experimentation and research.

---


