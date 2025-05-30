# Chapter 31: Image Segmentation

> “*If object detection draws boxes, image segmentation draws boundaries.*”

---

## What Is Image Segmentation?

Image Segmentation is the process of classifying each pixel in an image into a category. Instead of just identifying or locating objects, segmentation understands the exact shape and boundary of each object.

There are two main types:

|Type	                |Description                                                                            |
|-----------------------|---------------------------------------------------------------------------------------|
|Semantic Segmentation	|Classifies every pixel (e.g., "this pixel is part of a cat")                           |
|Instance Segmentation	|Distinguishes between multiple objects of the same class (e.g., "cat #1" vs "cat #2")  |

---

## Applications

- **Autonomous Vehicles**: Detect roads, pedestrians, signs

- **Medical Imaging**: Tumor segmentation in MRIs/CTs

- **Agriculture**: Identify plant species in satellite images

- **Robotics**: Scene understanding for manipulation

---

## Dataset Example: Oxford Pets

The [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) includes:

- Images of pets (cats/dogs)  
- Segmentation masks where each pixel is labeled as pet, background, or border

TensorFlow Datasets provides a wrapper:
```python
import tensorflow_datasets as tfds

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
```

---

## Building a Segmentation Model (U-Net Style)

### Step 1: Preprocess Input & Labels
```python
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.uint8)
    return input_image, input_mask
```

---

### Step 2: U-Net Architecture (Encoder–Decoder)

```python
from tensorflow.keras import layers, models

def unet_model(output_channels):
    inputs = layers.Input(shape=[128, 128, 3])

    # Encoder
    down1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    down1 = layers.MaxPooling2D()(down1)

    down2 = layers.Conv2D(128, 3, activation='relu', padding='same')(down1)
    down2 = layers.MaxPooling2D()(down2)

    # Bottleneck
    bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')(down2)

    # Decoder
    up1 = layers.UpSampling2D()(bottleneck)
    up1 = layers.Concatenate()([up1, down2])
    up1 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)

    up2 = layers.UpSampling2D()(up1)
    up2 = layers.Concatenate()([up2, down1])
    up2 = layers.Conv2D(64, 3, activation='relu', padding='same')(up2)

    outputs = layers.Conv2D(output_channels, 1, activation='softmax')(up2)

    return models.Model(inputs=inputs, outputs=outputs)
```

---

### Step 3: Train the Model
```python
model = unet_model(output_channels=3)  # background, object, border

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_batches, epochs=20, validation_data=val_batches)
```

---

## Visualize the Segmentation Output

```python
import matplotlib.pyplot as plt

def display(display_list):
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
```

---

## Evaluation Tips

- Use **IoU** (Intersection over Union) for pixel-level accuracy  
- Use **Dice coefficient** for imbalanced segmentation tasks  
- Visually inspect masks to verify semantic alignment

---

## Try It Live

🔗 [Colab: U-Net Semantic Segmentation](https://colab.research.google.com/)  
(Coming soon to the repo: semantic segmentation playground with custom uploads!)

---

## Summary

In this chapter, you:

- Understood the difference between classification, detection, and segmentation  
- Built a U-Net-based semantic segmentation model in TensorFlow  
- Visualized how the model learns to identify object boundaries, pixel by pixel

---

