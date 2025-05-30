# Chapter 21: Text Preprocessing & Tokenization

> “*In the world of language, meaning is hidden in tokens. Before a model can read, we must teach it to parse.*”

---

Natural Language Processing (NLP) starts with text, but machines only understand numbers. This chapter introduces the crucial preprocessing steps needed to convert raw language into model-ready tensors.

You’ll learn:

- Why text preprocessing is necessary  
- How tokenization works in TensorFlow  
- How to handle sequences of different lengths  
- The role of padding, truncation, and vocab management  
- How to build a Tokenizer pipeline using tf.keras and TextVectorization

By the end, you’ll be ready to feed text into deep learning models.

---

## Why Preprocessing Matters

- Text is messy:  
- Varying lengths  
- Punctuation, emojis, casing  
- Slang, typos, multiple languages

We must normalize and vectorize text into a format that TensorFlow can understand.

---

## Step 1: Text Normalization

Standard steps include:

- Lowercasing  
- Removing punctuation  
- Handling contractions or special characters

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
```

But TensorFlow offers a cleaner way via `TextVectorization`.

---

## Step 2: Tokenization with `TextVectorization`

```python
from tensorflow.keras.layers import TextVectorization

# Example data
texts = tf.constant(["TensorFlow is great!", "Natural Language Processing."])

# Create vectorizer
vectorizer = TextVectorization(
    max_tokens=1000,
    output_mode='int',
    output_sequence_length=6
)

vectorizer.adapt(texts)
vectorized = vectorizer(texts)
print(vectorized)
```
**Output:**
Each sentence is converted into a fixed-length sequence of integers (tokens).

---

## Sequence Padding

Neural networks (like RNNs or Transformers) require fixed-length inputs.

If using `Tokenizer` + `pad_sequences` manually:
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=6, padding='post')
```

---

## Padding Options

|Argument	    |Options	            |Description                            |
|---------------|-----------------------|---------------------------------------|
|padding	    |'pre', 'post'	        |Add zeros before or after sequence     |
|truncating	    |'pre', 'post'	        |Cut long sequences before or after     |
|maxlen	        |int	                |Final sequence length after padding    |

---

## View Token-to-Word Mapping

```python
word_index = tokenizer.word_index
print(word_index['tensorflow'])  # Shows token assigned to word
```

---

## Dealing with Out-of-Vocab (OOV)

Words not seen during f`it_on_texts()` are mapped to a special token (e.g., `<OOV>`).

Ensure you define `oov_token='<OOV>'` to handle unseen text gracefully during inference.

---

## Word Embeddings (Preview)

Once tokenized, you can pass sequences into an Embedding layer:
```python
from tensorflow.keras.layers import Embedding

embedding = Embedding(input_dim=1000, output_dim=16, input_length=6)
output = embedding(vectorized)
```
This converts token indices into dense, learnable vector representations.

---

## Summary

In this chapter, you:

- Learned to clean and normalize raw text  
- Used TextVectorization for modern tokenization  
- Managed sequence length via padding and truncation  
- Prepared inputs for embeddings and models

Text preprocessing is the foundation of any NLP pipeline. With clean, consistent input, models can focus on learning patterns—not battling messy data.

---

