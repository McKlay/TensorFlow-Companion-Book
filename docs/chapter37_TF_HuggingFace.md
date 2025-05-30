# Chapter 37: Integrating TensorFlow with Hugging Face

> “*When TensorFlow meets Hugging Face, models gain both muscle and memory.*”

---

## Introduction: The Best of Both Worlds

- TensorFlow is an industrial-grade engine. Hugging Face is a model zoo with superpowers. Combine the two and you get:

- Access to thousands of pretrained models (BERT, GPT-2, ViT, T5, etc.)

- Seamless tokenization, loading, and fine-tuning

- Exportable `.h5`, `.pb`, or `.tflite` models for real-world deployment

In this chapter, we’ll walk through how to use Hugging Face Transformers with TensorFlow, including:

- Loading pretrained models

- Tokenizing data

- Fine-tuning for NLP tasks

- Saving & exporting for inference

---

## Step 1: Install the Libraries

```bash
pip install -q transformers datasets
```

---

## Step 2: Load a Pretrained BERT Model

Let’s classify sentiment using `bert-base-uncased`.

```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

---

## Step 3: Prepare Data (e.g., IMDb Sentiment)

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"][:3000]
train_labels = dataset["train"]["label"][:3000]

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='tf')
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).batch(16)
```

---

##  Step 4: Compile and Fine-Tune

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.fit(train_dataset, epochs=2)
```

---

## Step 5: Save and Export

Save the model (TensorFlow format)
```python
model.save_pretrained("bert-finetuned-imdb")
tokenizer.save_pretrained("bert-finetuned-imdb")
```

You can now:

- Reload it via `TFAutoModelForSequenceClassification.from_pretrained()`

- Convert to `.tflite` using `TFLiteConverter`

- Deploy on Hugging Face Spaces or use `transformers.pipelines` for inference

---

## Bonus: Quick Prediction with `pipeline()`

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="bert-finetuned-imdb", tokenizer="bert-finetuned-imdb")
print(classifier("I loved this movie! So well-acted and emotional."))
```

---

## Summary

In this chapter, you learned:

- How to load and fine-tune Hugging Face models using TensorFlow

- How to tokenize text and structure datasets for training

- How to export models and deploy them in real-world apps

TensorFlow’s maturity in deployment combined with Hugging Face’s innovation in pretrained models gives you the **ultimate toolkit** — powerful models with plug-and-play simplicity.

Whether you’re building sentiment analyzers, question-answering bots, or vision transformers, this integration helps you **stand on the shoulders of AI giants**.

---

