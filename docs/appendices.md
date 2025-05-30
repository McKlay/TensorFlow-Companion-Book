# Appendices

> “*The end of the book is the beginning of mastery.*”

---

These appendices are designed to serve as your quick-reference toolkit and troubleshooting companion as you build real-world TensorFlow projects.

---

## Appendix A: Tensor Shapes Cheat Sheet (TensorFlow Style)

|Shape Notation	                            |Meaning	                            |Example            |
|-------------------------------------------|---------------------------------------|-------------------|
|`(batch_size,)`	                        |1D vector (e.g., labels)	            |`[32]`             |
|`(batch_size, features)`	                |2D input (e.g., tabular data)	        |`[32, 10]`         |
|`(batch_size, height, width, channels)`	|Image input (NHWC format)	            |`[32, 28, 28, 1]`  |
|`(batch_size, time_steps, features)	`   |Sequences (e.g., RNN input)	        |`[32, 100, 64]`    |
|`(vocab_size, embedding_dim)`	            |Word embeddings	                    |`[10000, 300]`     |

**💡 Did you know?**
TensorFlow prefers **NHWC** format for convolutional layers: [**batch**, **height**, **width**, **channels**], while PyTorch uses **NCHW** by default.

---

## Appendix B: TensorFlow vs PyTorch – Code Comparison

|Task	                |TensorFlow (Keras)	                    |PyTorch                                |
|-----------------------|---------------------------------------|---------------------------------------|
|Dense Layer	        |`tf.keras.layers.Dense(128)`	        |`nn.Linear(in, out)`                     |
|Activation	            |`activation='relu'`	                |`F.relu(x)`                              |
|Loss Function	        |`SparseCategoricalCrossentropy()`	    |`nn.CrossEntropyLoss()`                  |
|Optimizer	            |`Adam(learning_rate=1e-3)`	            |`optim.Adam(model.parameters())`         |
|Training Loop	        |`model.fit(x, y)`	                    |`for epoch in ...: optimizer.step()`     |
|Model Definition	    |Subclass `tf.keras.Model`	            |Subclass `nn.Module`                     |

---

## Appendix C: Debugging Tips for TensorFlow

|Issue	                                        |Cause	                                    |Fix                                            |
|-----------------------------------------------|-------------------------------------------|-----------------------------------------------|
|`Graph execution error`	                    |Mixing eager + graph mode	                |Use `@tf.function` carefully                   |
|`ValueError: Shapes (None, 1) != (None, )`	    |Shape mismatch	                            |Check model output vs label shape              |
|Model not learning	                            |Wrong loss, optimizer, or learning rate	|Try `lr=1e-3` and experiment                   |
|GPU not being used	                            |Device misconfiguration	                |Use `tf.config.list_physical_devices('GPU')`   |
|Out-of-memory error (OOM)	                    |Batch size too large or model too big	    |Reduce batch size or use mixed precision       |

---

## 🗂 Appendix E: Full API Reference Crosswalk

> “*Knowing what’s possible is the first step to mastery.*”

---

This appendix maps the most essential TensorFlow API classes and functions to their official documentation links, along with a quick summary of when and why you’d use each.

|Module	                        |API	                                            |Summary	                                                                         |Docs
|-------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
|tf.Tensor	                    |tf.constant, tf.Variable	                        |Core tensor creation. Use constants for fixed data, variables for trainable parameters.	tf.Tensor                                                            |[tf.tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor)                 |
|tf.keras	                    |Sequential, Model, Layer	                        |Core model-building classes. Sequential for stacks, subclass Model for custom.	tf.keras                                                                               |[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)                   |
|tf.keras.layers	            |Dense, Conv2D, LSTM, Flatten, Dropout	            |Building blocks of neural networks: fully connected, CNN, RNN, etc.	tf.keras.layers                                                                              |[tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)     |
|tf.keras.losses	            |CategoricalCrossentropy, MSE,                      |Huber	Standard loss functions for classification, regression, and robust fitting.	tf.keras.losses                                                                     |[tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses)     |
|tf.keras.optimizers	        |Adam, SGD, RMSprop	                                |Optimizers for gradient descent-based training.	tf.keras.optimizers                                                                          |[tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)                                                                                                                                                            |
|tf.data	                    |Dataset.from_tensor_slices, .batch(), .shuffle()	|Efficient, scalable data pipelines for training.	tf.data                                                                                |[tf.data](https://www.tensorflow.org/api_docs/python/tf/data)                     |
|tf.image	                    |resize, random_flip, per_image_standardization	    |Image preprocessing utilities (augmentation, normalization).	tf.image                                                                               |[tf.image](https://www.tensorflow.org/api_docs/python/tf/image)                   |
|tf.function	                |@tf.function	                                    |Converts Python code into efficient graph-based TensorFlow execution.	tf.function                                                                            |[tf.function](https://www.tensorflow.org/api_docs/python/tf/function)             |
|tf.GradientTape	            |tf.GradientTape()	                                |Enables automatic differentiation for custom training loops.	tf.GradientTape                                                                        |[tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape)     |
|tf.lite	                    |TFLiteConverter, Interpreter	                    |Tools to convert and run models on mobile/embedded devices.	tf.lite                                                                                |[tf.lite](https://www.tensorflow.org/lite/guide)                                  |
|tensorflow_recommenders	    |tfrs.Model, tfrs.tasks.Retrieval	                |TensorFlow Recommenders for personalized ranking and retrieval.	TensorFlow Recommenders                                                                        |[TensorFlow Recommmenders](https://www.tensorflow.org/recommenders)               |
|transformers (HF)	            |TFAutoModel, AutoTokenizer, pipeline()	            |Hugging Face Transformers (TF version) for NLP & vision models.	Hugging Face Transformers                                                                        |[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)       |
|tfx	                        |ExampleGen, Trainer, Pusher, Evaluator	            |TensorFlow Extended pipeline components for production ML.	TFX                                                                                 |[TFX](https://www.tensorflow.org/tfx/guide)                                       |
|tensorboard	                |tf.summary, TensorBoard callback	                |Logs scalars, images, and graphs during training.	TensorBoard                                                                         |[TensorBoard](https://www.tensorflow.org/tensorboard/get_started)                 |

---

#### 💡 Pro Tip: You can quickly access any TensorFlow documentation by appending the API name to:
📎 [https://www.tensorflow.org/api_docs/python/tf/**YourAPIHere**](https://www.tensorflow.org/api_docs/python/tf)
