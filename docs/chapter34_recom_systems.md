# Chapter 34: Recommender Systems

> “*The best suggestions are those you didn’t even know you needed — until the machine learned them.*”

---

## 🎬 Introduction: Behind Every “You May Also Like…”

From Netflix suggesting your next binge to Amazon nudging your next impulse buy, recommender systems are the engine behind personalized digital experiences. They model your behavior, compare it with others, and predict what you’ll likely enjoy next.

In this chapter, we’ll explore how to build recommender systems in TensorFlow — starting from the simplest collaborative filtering techniques to deep learning-powered ranking models using the tf.recommendation ecosystem and TensorFlow Recommenders (TFRS).

---

## Types of Recommender Systems

There are three core approaches:

- **Content-Based Filtering**  
Recommends items similar to ones the user liked (based on item features)

- **Collaborative Filtering**  
Recommends based on user-item interactions (e.g., matrix factorization)

- **Hybrid Systems**  
Combine both worlds — Netflix is a classic example

---

## Dataset: MovieLens Mini

We'll use the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/) — a standard for testing recommender systems.
```python
!pip install -q tensorflow-recommenders
!pip install -q tensorflow-datasets

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import tensorflow as tf

ratings = tfds.load('movielens/100k-ratings', split='train')
movies = tfds.load('movielens/100k-movies', split='train')
```

---

## Preprocessing the Data

```python
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})

movie_titles = movies.map(lambda x: x["movie_title"])

# Vocabulary for embedding
user_ids = ratings.map(lambda x: x["user_id"])
movie_titles = ratings.map(lambda x: x["movie_title"])

unique_user_ids = np.unique(list(user_ids.as_numpy_iterator()))
unique_movie_titles = np.unique(list(movie_titles.as_numpy_iterator()))
```

---

## Building the Model

### 1. User & Movie Embedding Models
```python
embedding_dim = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)
])

movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
    tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dim)
])
```

---

## 2. Metric Learning Model with TFRS
```python
class MovieModel(tfrs.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movie_titles.batch(128).map(movie_model)
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)
```

---

## Training

```python
model = MovieModel(user_model, movie_model)
model.compile()

cached_ratings = ratings.shuffle(100_000).batch(8192).cache()

model.fit(cached_ratings, epochs=3)
```

---

## Making Recommendations

```python
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movie_titles.batch(100).map(lambda title: (title, model.movie_model(title)))
)

# Predict for a user
user_id = "42"
scores, titles = index(tf.constant([user_id]))
print(f"Recommendations for user {user_id}: {titles[0, :3].numpy()}")
```

---

## Summary

In this chapter, you learned:

- The difference between collaborative, content-based, and hybrid recommendation systems  
- How to use the TensorFlow Recommenders (TFRS) library  
- How to train an end-to-end retrieval model using user and item embeddings  
- How to generate personalized recommendations

Recommender systems are deeply embedded in modern software ecosystems — from e-commerce to social media. By learning how to build one, you gain the ability to create truly personalized user experiences.

---

