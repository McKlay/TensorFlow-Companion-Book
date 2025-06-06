---
hide:
  - toc
---

# Chapter 23: RNNs & LSTMs

> “*Words are not isolated—they remember what came before. RNNs give models a sense of time.*”

---

Words in a sentence form a sequence—and meaning often depends on order. Traditional models like Bag-of-Words or TF-IDF treat words as independent, which limits their ability to capture structure or context.

Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) units were among the first neural architectures designed to remember context over time—a fundamental shift in how machines processed text.

By the end of this chapter, you’ll:

- Understand how RNNs and LSTMs work  
- Use tf.keras.layers.SimpleRNN and LSTM  
- Train a model on sequential data (e.g., text sentiment)  
- Visualize how memory affects predictions

---

## What Is an RNN?

RNNs process sequences one element at a time, passing hidden state from one time step to the next.
```text
Input:      I → love → TensorFlow
Hidden:    h0 → h1 → h2
Output:     y0 → y1 → y2
```
Each output depends not just on the current input, but also on past context.

---

## The Problem: Vanishing Gradients

RNNs struggle with long-term dependencies—earlier words lose influence as the sequence grows. That’s where LSTMs come in.

---

## LSTMs: Memory with Gates

LSTMs introduce internal cell states and gates (forget, input, output) to regulate information flow.

- Forget Gate: What to forget from the previous cell  
- Input Gate: What new information to store  
- Output Gate: What to pass to the next step  
- This design helps retain useful context across longer sequences.

---

## Implementing an LSTM in TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## Dataset: IMDB Sentiment (Binary)

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)
```

**Train the Model**
```python
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
```

---

## Visualizing Memory

Use the LSTM output layer to view how sentiment builds over time. You can create attention overlays or extract intermediate states:
```python
intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)
lstm_output = intermediate_model.predict(x_test[:1])
```

---

## When to Use RNNs / LSTMs

|Use Case	                |Recommended Model             |
|---------------------------|------------------------------|
|Short sequences	        |RNN                           |
|Long-range memory	        |LSTM or GRU                   |
|Streaming data	            |RNN/LSTM                      |
|Parallelization needed	    |Transformer (Next Chapter)    |

---

## GRU: Simpler Alternative
Gated Recurrent Unit (GRU) is a simplified version of LSTM:

- Combines forget and input gates  
- Fewer parameters, faster to train

```python
from tensorflow.keras.layers import GRU
model = Sequential([Embedding(10000, 64), GRU(128), Dense(1, activation='sigmoid')])
```

---

## Summary

In this chapter, you:

- Learned how RNNs and LSTMs retain sequential memory  
- Implemented an LSTM for text sentiment classification  
- Understood their advantages and limitations  
- Explored GRU as a lightweight alternative

RNNs and LSTMs were the backbone of NLP before Transformers. They taught us that order and memory matter in language.

---



