# Chapter 28: Image Classification

> “*At the end of the pipeline, it’s about making a decision—cat or dog, plane or car, tumor or healthy tissue.*”

---

## What Is Image Classification?

Image classification is the task of assigning a label from a predefined set of categories to an input image. Think:

- Is this image a cat or a dog?  
- Does this scan show pneumonia or not?  
- Which of the 10 digits does this handwritten number represent?

It’s one of the simplest yet most powerful applications of CNNs.

---

## Building an End-to-End Classifier with TensorFlow

Let’s classify images from CIFAR-10, a dataset of 60,000 32x32 color images in 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.

---

### Step 1: Load & Preprocess the Dataset

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0
```

---

### Step 2: Define the Model Architecture
 
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])
```

---

###  Step 3: Compile and Train

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

---

### Step 4: Evaluate the Model

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")
```

---

### Step 5: Make Predictions

```python
import numpy as np

predictions = model.predict(x_test)
print("Predicted label:", np.argmax(predictions[0]))
```
You can match the result to CIFAR-10’s label map:
```python
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
```

---

## Add Data Augmentation (Optional)

To improve generalization:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    ...
])
```

---

## Metrics to Monitor

|Metric	                |Why it matters                     |
|-----------------------|-----------------------------------|
|Accuracy	            |Basic measure for classification   |
|Confusion Matrix	    |Understand misclassified classes   |
|Precision/Recall	    |Especially for imbalanced classes  |

---

## Try It Live

🔗 [Colab Notebook: CIFAR-10 Classifier](https://colab.research.google.com/)  
(To be linked in the companion repo)

---

## Summary

In this chapter, you built a complete image classification model using CNNs. You learned to:

- Preprocess data and normalize images  
- Stack convolutional layers effectively  
- Compile, train, evaluate, and make predictions with TensorFlow 

---



