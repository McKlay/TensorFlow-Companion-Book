---
hide:
  - toc
---

# Chapter 29: Object Detection

> “*Image classification tells you what—object detection tells you where.*”

---

## What Is Object Detection?  

Object detection not only classifies what is in an image, but also where it is—by drawing bounding boxes around detected objects.

This enables applications like:

- Face detection in photos  
- Pedestrian detection for self-driving cars  
- Barcode readers in retail  
- Real-time object tracking in surveillance systems

---

## Classification vs Detection

|Task	                |Input	    |Output                 |
|-----------------------|-----------|-----------------------|
|Image Classification	|Image	    |Label (e.g., "dog")    |
|Object Detection	    |Image	    |Label + Bounding Box   |

Example:

**Classification**: “There is a cat in the image.”

**Detection**: “There is a cat at coordinates (x1, y1) to (x2, y2).”

---

## Tools for Object Detection in TensorFlow

### Option 1: Pretrained Models via `TensorFlow Hub`
```python
import tensorflow as tf
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/google/fasterrcnn/openimages_v4/inception_resnet_v2/1")
```

Input an image:

```python
import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).resize((512, 512))
    return np.array(img) / 255.0

image = load_image('sample.jpg')
result = detector(tf.convert_to_tensor([image]), training=False)
```

---

### Option 2: TF Object Detection API (Advanced)

TensorFlow has a dedicated [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) with models like:

- SSD (Single Shot MultiBox Detector)  
- Faster R-CNN  
- EfficientDet

⚠️ Requires setup with:

- Protobuf compilation  
- COCO-trained checkpoints  
- Custom label maps

---

## Using a Pretrained SSD Model 

For simpler use cases, try SSD from `tf.keras.applications` or prepackaged TF Hub models.
```python
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
```

Visualize predictions:
```python
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

image = tf.image.convert_image_dtype(tf.io.decode_jpeg(tf.io.read_file('cat.jpg')), tf.float32)
image_resized = tf.image.resize(image, (320, 320))

results = detector(tf.expand_dims(image_resized, axis=0))
print(results['detection_boxes'])
```

---

## Visualizing Bounding Boxes

```python
import matplotlib.patches as patches

def draw_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        y1, x1, y2, x2 = box
        rect = patches.Rectangle((x1*image.shape[1], y1*image.shape[0]), 
                                 (x2 - x1)*image.shape[1], 
                                 (y2 - y1)*image.shape[0], 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    plt.show()
```

---

## Real-World Applications

|Industry	        |Application                        |
|-------------------|-----------------------------------|
|Automotive         |Pedestrian/vehicle detection       |
|Retail	            |Checkout-free shopping             |
|Healthcare	        |Tumor boundary detection           |
|Manufacturing	    |Faulty component identification    |

---

##  Try It Live
🔗 [Colab Notebook: Object Detection with TF Hub](https://colab.research.google.com/)  
(Ready to be plugged into the companion repo)

---

## Summary

In this chapter, you:

- Learned how object detection adds spatial awareness to classification  
- Explored TensorFlow Hub and the Object Detection API  
- Built a pipeline to detect objects and visualize bounding boxes

---
