---
hide:
  - toc
---

# Chapter 22: TF-IDF, Bag-of-Words Representations

> “*Before transformers, we counted words. And in that counting, meaning emerged.*”

---

Before deep learning, NLP was powered by statistical representations of text—primarily Bag-of-Words (BoW) and TF-IDF.

These techniques are still useful today:

- They’re fast, interpretable, and effective on small datasets  
- Often used as baselines before training heavy models  
- Key to understanding how NLP evolved into embedding-based approaches

In this chapter, you'll learn:

- How to convert text into BoW and TF-IDF vectors  
- When to use count-based vs frequency-based methods  
- How to implement these using `TfidfVectorizer` and `CountVectorizer` from `scikit-learn`  
- How to visualize feature importance

## Bag-of-Words (BoW)

BoW represents text by counting how often each word appears, ignoring order.

Example:
```python
Doc 1: "TensorFlow is great"
Doc 2: "TensorFlow and Keras"

Vocabulary: ['and', 'great', 'is', 'keras', 'tensorflow']

Vectors:
Doc 1 → [0, 1, 1, 0, 1]
Doc 2 → [1, 0, 0, 1, 1]
```

---

## Using CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["TensorFlow is great", "TensorFlow and Keras"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

##  TF-IDF: Term Frequency–Inverse Document Frequency

TF-IDF downweights common words (like "is", "the") and upweights rare but important ones.

**Formula**:  

- TF: Number of times a word appears in a document  
- IDF: Inverse of how many documents contain that word

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```
Each number now reflects importance, not just frequency.

---

## When to Use What?

|Use Case	                            |Technique          |
|---------------------------------------|-------------------|
|Small text datasets	                |TF-IDF or BoW      |
|Quick baseline for classification	    |TF-IDF             |
|Sparse features, interpretable	        |BoW                |
|Neural network input	                |Embedding          |

---

## Optional: Visualize TF-IDF Features

```python
import pandas as pd

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df)
```
Use this to:

- Identify key words driving predictions  
- Visualize feature distributions

---

## Limitations

- Doesn’t consider word order ("not good" vs "good")  
- Vocabulary must be fixed  
- Large vocab = sparse vectors  
- Not suitable for capturing contextual meaning

That’s where deep learning models (like embeddings, RNNs, Transformers) step in.

---

## Summary

In this chapter, you:

- Learned the intuition behind BoW and TF-IDF  
- Used CountVectorizer and TfidfVectorizer in scikit-learn  
- Compared their strengths and limitations  
- Set the stage for embedding-based and deep NLP models

Classic vectorization methods are not obsolete—they’re foundational, fast, and still relevant in pipelines and search systems.


