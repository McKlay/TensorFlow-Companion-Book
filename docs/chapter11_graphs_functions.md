---
hide:
  - toc
---

# Chapter 11: Graphs & Functions (@tf.function)

> “*First you write Python. Then you write TensorFlow. Then you make Python act like TensorFlow.*”

---

## 11.1 Why Graphs?

Python is great for prototyping. But it’s slow when you scale to:  
- Production inference  
- Deployment across devices  
- Multi-GPU/TPU execution  
- Model export for TFLite or TensorFlow Serving

TensorFlow solves this with graphs: dataflow representations that can be optimized and executed outside the Python runtime.

---

## 11.2 Enter @tf.function

You write regular Python—but TensorFlow traces it once and converts it into a graph.

✅ Example:
```python
import tensorflow as tf

@tf.function
def compute(x):
    return x**2 + 3*x + 1

print(compute(tf.constant(2.0)))  # tf.Tensor(11.0, shape=(), dtype=float32)
```
This runs like Python but is compiled under the hood.

---

## 11.3 Benefits of Using @tf.function

- Faster execution: Runs as a graph instead of interpreted Python  
- Cross-platform compatibility: Can run on GPUs, TPUs, mobile, etc.  
- Serialization: Enables saving models in SavedModel format  
- Deployment: Used in TensorFlow Serving, TFLite, and TF.js

---

## 11.4 Gotchas & Debugging Tips

❗ Be careful of Python-side effects:
```python
@tf.function
def bad_func():
    print("This won't show up")  # Runs only during tracing, not each call
```
Only TensorFlow ops are tracked. Use `tf.print()` instead of Python `print()`:
```python
@tf.function
def good_func(x):
    tf.print("Value of x:", x)
```

---

## 11.5 Input Signatures (Optional)

Restrict the function to a fixed input type and shape for optimization:
```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def model(x):
    return tf.reduce_sum(x)
```
This makes tracing more predictable and speeds up model serving.

---

## 11.6 Retrieving the Graph

You can inspect the computation graph like this:
```python
@tf.function
def square(x):
    return x * x

print(square.get_concrete_function(tf.constant(2.0)).graph.as_graph_def())
```
This shows you the internal graph ops (used for debugging, export, or tooling).

---

## 11.7 Common Use Cases

|Use Case	                    |Benefit                    |
|-------------------------------|---------------------------|
|Training loops	                |Speed boost                |
|Model export (SavedModel)	    |Required                   |
|TF Serving / deployment	    |Required                   |
|Writing reusable pipelines	    |Cleaner graph structure    |

---

## 11.8 Summary  

- @tf.function converts Python code into high-performance TensorFlow graphs.  
- It enables speed, portability, deployment, and tracing.  
- Use tf.print instead of regular print() inside graph code.  
- It’s a must for production workflows, but test in eager mode first.  

---

> “*First you write Python. Then you write TensorFlow. Then you make Python act like TensorFlow.*”