---
hide:
  - toc
---

# Chapter 25: NLP Projects — Spam Detection, Sentiment Analysis, Autocomplete

> “*Language is powerful. Teaching machines to understand it unlocks infinite possibilities.*”

---

With your knowledge of tokenization, vectorization, RNNs, and Transformers, you're now ready to build end-to-end NLP applications. In this chapter, we’ll walk through three mini-projects using TensorFlow:

1. Spam Detection – Binary classification using classic vectorization
2. Sentiment Analysis – Text-to-emotion prediction using LSTM
3. Autocomplete – Next-word prediction using a Transformer

Each project includes:

- Data loading  
- Text preprocessing  
- Model architecture  
- Training loop  
- Evaluation and usage

---

## 1. Spam Detection with TF-IDF

**Dataset: SMS Spam Collection (UCI)**
```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
```

**Preprocessing with TfidfVectorizer**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = TfidfVectorizer(max_features=1000).fit_transform(df['message']).toarray()
y = df['label'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Model**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(1000,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

## 😊 2. Sentiment Analysis with LSTM

**Dataset: IMDB Reviews**
```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)
```

**Model**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=200),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

## 3. Autocomplete with Mini Transformer

This is a simplified version, using a small vocabulary.

**Sample Corpus**
```python
sentences = [
    "hello how are you",
    "hello how is your day",
    "i love natural language processing",
    "tensorflow is powerful"
]
```

**Vectorize & Build Transformer**

Use TextVectorization and custom Transformer layers (as built in Chapter 24). Train it as a language model: for every input sequence, predict the next word.

---

##  Evaluation Tips

|Project	            |Metric	            |Evaluation Example                     |
|-----------------------|-------------------|---------------------------------------|
|Spam Detection	        |Accuracy, F1	    |Confusion matrix                       |
|Sentiment Analysis	    |Accuracy	        |Predict reviews manually               |
|Autocomplete	        |Top-k accuracy	    |“hello how” → “are”, “is”, “you”       |

Use TensorBoard to visualize training curves, and checkpoints to save your models.

---

## Deployment Ideas

|Task	           |Deploy As                   |
|------------------|----------------------------|
|Spam Detection	   |FastAPI REST API            |
|Sentiment Model   |Hugging Face + Gradio       |
|Autocomplete	   |TensorFlow Lite on mobile   |

All three can be integrated into websites, apps, or backend systems for real-time inference.

---

## Summary

In this chapter, you:

- Built three full NLP projects from start to finish  
- Used both traditional ML and deep learning  
- Combined text preprocessing, tokenization, and modeling  
- Prepared these models for deployment and real-world use

These projects are a stepping stone to larger systems: chatbots, translation, summarization, and beyond.

---