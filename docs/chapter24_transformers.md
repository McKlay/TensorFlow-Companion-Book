---
hide:
  - toc
---

# Chapter 24: Transformers in TensorFlow (from Scratch)

> “*Forget recurrence. Attention is all you need.*”

---

Transformers marked a revolution in deep learning, replacing RNNs and LSTMs as the new foundation of sequence modeling. Their secret? Self-attention—a mechanism that lets models focus on relevant parts of input, regardless of position.

In this chapter, you will:

- Understand the intuition behind attention and transformers  
- Explore how positional encoding enables order-awareness  
- Build a simplified transformer encoder block using TensorFlow  
- See how models like BERT and GPT are built on this architecture

---

## Why Transformers?

RNNs process sequences sequentially, limiting parallelism. Transformers process entire sequences in parallel, and use attention to model relationships between tokens.

🔁 From RNN to Transformer
```lua
RNN: word₁ → word₂ → word₃ → ...
Transformer: [word₁, word₂, word₃] all at once
```

---

## Self-Attention Explained

Self-attention allows the model to weigh the importance of different words when encoding a particular token.

> “The bank was crowded.”
> Self-attention helps distinguish: money bank vs river bank using context like “crowded.”

---

## Core Transformer Components

|Component	                |Purpose                                        |
|---------------------------|-----------------------------------------------|
|Embedding Layer	        |Convert tokens to dense vectors                |
|Positional Encoding	    |Add order information                          |
|Multi-Head Attention	    |Attend to different parts of the sequence      |
|Feed-Forward Network	    |Process attended representations               |
|Residual & LayerNorm	    |Improve stability and gradient flow            |

---

## Step-by-Step: Mini Transformer Encoder

**Step 1: Positional Encoding**
```python
import numpy as np
import tensorflow as tf

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads, dtype=tf.float32)
```

**Step 2: Scaled Dot-Product Attention**
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(d_k)
    
    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(weights, v)
```

**Step 3: Multi-Head Attention Layer**
```python
from tensorflow.keras.layers import Dense, Layer

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.num_heads = num_heads

        self.Wq = Dense(d_model)
        self.Wk = Dense(d_model)
        self.Wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)

        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled_attention, (batch_size, -1, self.num_heads * self.depth))
        return self.dense(concat)
```

---

## Transformer Encoder Block

```python
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense

class TransformerEncoderBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.att(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
```

---

## Training Transformer on Text (Preview)

You can now stack transformer blocks, add embeddings, and train using `model.fit()` as usual. Later chapters (esp. Part IV and V) show this for:

- Sentiment analysis  
- Text classification  
- Question answering

---

## BERT and GPT: Built on Transformers

|Model	|Direction	    |Use Case                   |
|-------|---------------|---------------------------|
|BERT	|Bidirectional	|Classification, QA         |
|GPT	|Left-to-right	|Generation, Autocomplete   |

You can fine-tune these using `transformers` from Hugging Face or build them from scratch (see next chapter for applications).

---

## Summary

In this chapter, you:

- Understood how attention replaces recurrence  
- Built a mini transformer encoder using TensorFlow  
- Explored the components like positional encodings and multi-head attention  
- Learned how modern models like BERT and GPT are architecturally composed

Transformers are the foundation of state-of-the-art NLP and many vision models. You’ve now walked through the gears that power them.

---