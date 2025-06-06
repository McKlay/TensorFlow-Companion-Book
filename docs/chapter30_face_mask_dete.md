---
hide:
  - toc
---

# Chapter 30: Face Mask Detection

> “*When vision meets responsibility—models that see and protect.*”

---

## 😷 Why Face Mask Detection?

The COVID-19 pandemic highlighted the need for **real-time face mask detection** in public safety systems. Detecting whether a person is wearing a mask can:

- Enhance compliance in public places  
- Assist automated access control  
- Provide insights for health monitoring systems

This task is a real-world application of **binary object classification** layered on top of **face detection**.

---

## Problem Breakdown

Detect faces in an image

Classify each face:

- `with_mask`  
- `without_mask`

This is different from simply classifying the whole image—it’s **region-specific classification**.

---

## Dataset

We’ll use the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset), which contains:

- Labeled images of people with and without masks  
- Optionally: bounding boxes (if using detection-first approach)

Alternatively, for simplicity, use a f**older-structured dataset**:

```lua
dataset/
  ├── with_mask/
  └── without_mask/
```

---

## Implementation Steps

### Step 1: Load and Preprocess Images

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
```

---

### Step 2: Build a CNN Model

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])
```

---

### Step 3: Compile and Train

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_gen, validation_data=val_gen, epochs=10)
```

---

### Step 4: Make Predictions

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('test.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
print("With mask" if prediction < 0.5 else "Without mask")
```

---

## Optional: Use OpenCV for Real-Time Detection

Integrate with your webcam:

```python
import cv2

# Load model and Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128,128)) / 255.0
        prediction = model.predict(np.expand_dims(face_resized, axis=0))[0][0]
        label = "With Mask" if prediction < 0.5 else "No Mask"
        color = (0,255,0) if prediction < 0.5 else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow('Face Mask Detector', frame)
    if cv2.waitKey(1) == 27: break  # Esc to exit

cap.release()
cv2.destroyAllWindows()
```

---

## Deployment Tip

Use:

- TensorFlow Lite for mobile deployment  
- FastAPI + OpenCV for web camera APIs  
- Hugging Face Spaces for demo apps

---

## Summary

In this chapter, you built a binary image classifier to detect face masks using TensorFlow and CNNs. You also explored how to:

- Train a model with folder-labeled images  
- Use real-time webcam feeds for predictions  
- Visualize and deploy your results

---


