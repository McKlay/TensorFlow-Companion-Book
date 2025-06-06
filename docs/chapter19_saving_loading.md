---
hide:
  - toc
---

# Chapter 19: Saving, Loading, and Callbacks

> “*Training is hard-earned. Never lose progress. Save smart, load fast, and train with foresight.*”

---

This chapter focuses on one of the most practical and essential aspects of any machine learning workflow: saving and restoring models and automating training behavior with callbacks.

By the end, you'll know how to:  

- Save your model weights or full architecture  
- Load and resume training or inference  
- Use callbacks like EarlyStopping, ModelCheckpoint, and custom logic  
- Ensure your work survives interruptions and scales reliably

## Saving Your Model

There are two main formats in TensorFlow:

#### 1. SavedModel Format (Recommended)
```python
model.save('saved_model/my_model')
```
To load:
```python
new_model = tf.keras.models.load_model('saved_model/my_model')
```
✅ Includes:

- Architecture  
- Weights  
- Training config  
- Optimizer state  

#### 2. HDF5 Format (.h5)
```python
model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
```
This format is widely used in other frameworks but has limitations with newer custom layers and optimizers.

---

## Saving Only Weights

If you just want to save parameters:
```python
model.save_weights('weights.ckpt')
```
To load them into the same architecture:
```python
model.load_weights('weights.ckpt')
```
Useful when:

- You want to share weights but not the model code  
- You're using subclassed models

---

## Using Callbacks
Callbacks are hooks into the training loop that let you intervene at key moments: epoch end, batch end, etc.

#### 🔹 Common Callbacks

**1. Early Stopping**
Stop training if validation loss doesn’t improve.
```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```
**2. Model Checkpoint**
Save model during training.
```python
tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
```
**3. Learning Rate Scheduler**
Adjust learning rate during training.
```python
tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)
```

---

## Using Callbacks in `fit()`

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', save_best_only=True)
]

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=20,
    callbacks=callbacks
)
```

---

## Custom Callback (Advanced)

You can define your own logic:
```python
class PrintEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} finished. Accuracy: {logs['accuracy']:.4f}")
```

Use with:
```python
model.fit(..., callbacks=[PrintEpoch()])
```

---

## Summary

In this chapter, you:

- Learned to save and load models in multiple formats  
- Used callbacks to control training and automate decisions  
- Understood when to save weights only vs full model  
- Built your own custom callback for fine-grained control

These tools are critical for long training runs, collaborative projects, and production deployment. Always train as if your power might go out. ⚡

---
