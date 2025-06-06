---
hide:
  - toc
---

# Chapter 4: Installation & Setup

> “A neural net’s journey begins with a single tensor.”

---

## 4.1 Preparing Your Workspace

Let’s keep things clean and self-contained. You’ll be using a virtual environment inside your TensorFlow folder for local experimentation.

✅ Step-by-step:

step 1. Navigate to your project folder  
```bash
    cd C:\Users\Clay\Desktop\Tutorials\TensorFlow
```
step 2. Create a virtual environment  
```bash
    python -m venv tf_env
```
step 3. Activate the environment  

- On CMD:
```bash
    .\tf_env\Scripts\activate
```
- On PowerShell:
```bash
    .\tf_env\Scripts\Activate.ps1
```
step 4. Upgrade pip & install TensorFlow (with GPU support)
```bash
pip install --upgrade pip
pip install tensorflow[and-cuda]
```
> ⚠️ This will install ~2.5 GB of GPU-enabled TensorFlow with pre-bundled CUDA & cuDNN (no manual install needed in TF 2.15+).

---

## 4.2 Verifying Installation & GPU Access
Create a file called `check_tf_gpu.py`:
```python
import tensorflow as tf

def print_gpu_info():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpus))
    for gpu in gpus:
        print("GPU Detected:", gpu.name)

if __name__ == '__main__':
    print_gpu_info()
```
Run it:
```bash
python check_tf_gpu.py
```

✅ Expected Output:
```yaml
TensorFlow version: 2.x.x
Num GPUs Available: 1
GPU Detected: NVIDIA GeForce RTX 4050 Laptop GPU
```
If it shows Num GPUs Available: 0, let’s talk. We riot. (But also debug your drivers or reinstall with CPU-only fallback.)

---

## 4.3 Bonus: Enable Dynamic GPU Memory Growth

Prevent TensorFlow from hoarding all your GPU VRAM upfront:
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on GPU.")
    except RuntimeError as e:
        print(e)
```
Use this in training scripts to allocate GPU memory only as needed.

---

## 4.4 Optional: Freeze Your Environment

To create a portable list of all packages:
```bash
pip freeze > requirements.txt
```
Useful when sharing your book repo or collaborating with others.

---

“A neural net’s journey begins with a single tensor.”


