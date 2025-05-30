# Chapter 8: Ragged, Sparse, and String Tensors

> “*Not all data fits in neat boxes. TensorFlow still makes it work*.”

---

## 8.1 What Are Non-Standard Tensors?

Not all data comes in a clean matrix shape like [batch_size, features]. Real-world examples often include:  

- Sentences of different lengths (NLP)
- Feature vectors with missing values
- Text tokens, file paths, categorical strings

Enter:  

- tf.RaggedTensor
- tf.SparseTensor
- tf.Tensor (with dtype=tf.string)

---

## 8.2 Ragged Tensors – For Variable-Length Sequences

### Use Case:
Think of sentences with different numbers of words:
```python
sentences = [
    ["Hello", "GPT-san"],
    ["TensorFlow"],
    ["Welcome", "to", "deep", "learning"]
]
```
### ✅ Code:
```python
import tensorflow as tf

rt = tf.ragged.constant([
    [1, 2, 3],
    [4, 5],
    [6]
])

print(rt)
print("Shape:", rt.shape)
```
###  Key Features:
- Use .ragged_rank to inspect hierarchy  
- Can still use many standard ops like indexing, slicing  
- Great for tokenized text or nested lists

---

## 8.3 Sparse Tensors – For Efficiency in Mostly-Zero Data

### Use Case:
When most values in a tensor are zero, storing all of them is wasteful. Use tf.SparseTensor to store just the non-zeros.

✅ Code:
```python
st = tf.sparse.SparseTensor(
    indices=[[0, 1], [1, 0]],
    values=[10, 20],
    dense_shape=[3, 3]
)

dense = tf.sparse.to_dense(st)
print(dense)
```
### Key Features:  
- Saves memory for large sparse data (e.g. recommender systems, one-hot vectors)
- Can convert to/from dense tensors
- Used heavily in embedding lookup and graph data

---

## 8.4 String Tensors – For Text Data

### Use Case:
NLP often starts with raw strings—TensorFlow supports them natively.
```python
str_tensor = tf.constant(["Tensor", "Flow", "Rocks"])
print(str_tensor)
print(tf.strings.length(str_tensor))       # Character length
print(tf.strings.upper(str_tensor))        # Uppercase conversion
print(tf.strings.join([str_tensor, "!"]))  # Add exclamations
```
### Key Features:  
- Native support for Unicode  
- Integrates with tf.strings, tf.text, and TextVectorization  
- First step before tokenization

---

## 8.5 Summary  
- Ragged tensors store data with uneven lengths (e.g. variable-length sentences).
- Sparse tensors store only non-zero elements—ideal for memory efficiency.
- String tensors enable natural language input processing natively.
- These types unlock real-world workflows where structure is messy or incomplete.

---

> “*Not all data fits in neat boxes. TensorFlow still makes it work*.”



