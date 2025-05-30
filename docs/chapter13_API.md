# Chapter 13: TensorFlow Keras API – Anatomy of a Model

> “*A model is just an idea—until it gets layers, weights, and shape.*”

---

## 13.1 What is `tf.keras`?

`tf.keras` is TensorFlow’s official high-level API for building, training, and deploying machine learning models.

It's designed to be:  

- User-friendly (simple syntax)  
- Modular (layers, optimizers, callbacks)  
- Extensible (custom layers/models)  
- Integrated (with TensorFlow ecosystem)  

> Keras wraps the complexity of TensorFlow so you can focus on structure and logic, not boilerplate.

---

## 13.2 The 3 Model Building Styles

There are three ways to build models using `tf.keras`:

### ✅ 1. Sequential API (Beginner-friendly)
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])
```

### ✅ 2. Functional API (Flexible architectures)
```python
inputs = tf.keras.Input(shape=(100,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### ✅ 3. Subclassing API (For full control)
```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.dense1(x)
        return self.out(x)

model = MyModel()
```
Each has trade-offs. Start with Sequential, move to Functional for branching inputs/outputs, and use Subclassing for full customization.

---

## 13.3 Anatomy of a Keras Model

Here’s what makes up a model under the hood:

|Component	        |Description                                    |
|-------------------|-----------------------------------------------|
|Input Layer	    |Defines the shape of input data                |
|Hidden Layers	    |The intermediate processing units              |
|Output Layer	    |Final layer for predictions                    |
|Loss Function	    |Measures model’s error                         |
|Optimizer	        |Updates weights based on gradients             |
|Metrics	        |Monitors performance (accuracy, loss, etc.)    |

---

## 13.4 Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
This sets the training configuration, including how the model learns and what it tracks.

---

## 13.5 Summary of a Simple Model Lifecycle

```python
# 1. Build the model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 2. Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train
model.fit(x_train, y_train, epochs=5)

# 4. Evaluate
model.evaluate(x_test, y_test)

# 5. Predict
preds = model.predict(x_new)
```

---

## 13.6 Summary  

- tf.keras is TensorFlow’s high-level API for model building.  
- You can build models using Sequential, Functional, or Subclassing styles.  
- Models have layers, losses, optimizers, and metrics—all handled cleanly.  
- Knowing the anatomy helps you debug, customize, and scale efficiently.  

---

“A model is just an idea—until it gets layers, weights, and shape.”