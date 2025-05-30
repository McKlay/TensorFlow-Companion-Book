# Chapter 32: GANs for Image Generation

> “*Two networks, locked in a dance. One imagines, the other critiques. Together, they create.*”

---

## What Are GANs?

**Generative Adversarial Networks (GANs)** are a class of deep learning models invented by **Ian Goodfellow** in 2014. Their purpose: **generate new**, **realistic** data from noise.

A **GAN** consists of two neural networks:

- **Generator (G)**: Learns to create realistic images  
- **Discriminator (D)**: Learns to tell real from fake images

These two models are trained in a **zero-sum** game:

- The Generator tries to **fool** the Discriminator  
- The Discriminator tries to **catch** the Generator’s fakes

Over time, both improve until the generated images are indistinguishable from real ones.

---

## GAN Architecture Overview
```lua
Noise (z) → Generator → Fake Image
                     ↓
        Discriminator ← Real Image
```

|Component	        |Purpose                                |
|-------------------|---------------------------------------|
|Generator	        |Converts random noise into images      |
|Discriminator	    |Judges real vs fake images             |
|Training	        |Minimax game until equilibrium         |

---

## Dataset: MNIST for Simplicity

We’ll use the MNIST dataset of handwritten digits for training a simple GAN.

```python
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
```

---

## Build the Generator

```python
from tensorflow.keras import layers

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    ])
    return model
```

---

## Build the Discriminator

```python
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model
```

---

## Training the GAN

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Loss for discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Loss for generator
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

## Optimizers

```python
generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

---

## Training Loop

```python
import time

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))
```

---

## Visualizing Generated Images

```python
import matplotlib.pyplot as plt

def generate_and_plot_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.show()
```

---

## Try It Live

🔗 [Colab: MNIST GAN from Scratch](https://colab.research.google.com/)  
(Will be added soon, to my companion repo for users to play with and train their own generators!)

---

## Summary

In this chapter, you learned how GANs:

- Work through adversarial training of Generator and Discriminator  
- Can generate realistic images from noise  
- Are implemented step-by-step using TensorFlow

GANs are a bridge between art and intelligence—used in face generation, style transfer, image-to-image translation, super-resolution, and more.

---


