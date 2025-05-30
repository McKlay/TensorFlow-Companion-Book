![GitHub last commit](https://img.shields.io/github/last-commit/McKlay/pytorch-companion-book)
![GitHub Repo stars](https://img.shields.io/github/stars/McKlay/pytorch-companion-book?style=social)
![GitHub forks](https://img.shields.io/github/forks/McKlay/pytorch-companion-book?style=social)
![MIT License](https://img.shields.io/github/license/McKlay/pytorch-companion-book)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=McKlay.pytorch-companion-book)



# PyTorch Builder’s Companion Book

> A human-readable, API-structured technical book that explores the core of `torch` — the beating heart of PyTorch.

**Live Site**: [https://mcklay.github.io/pytorch-companion-book/](https://mcklay.github.io/pytorch-companion-book/)  
Author: [Clay Mark Sarte](https://github.com/McKlay)  
Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) | Powered by [PyTorch](https://pytorch.org)

---

## What This Is

The **PyTorch Builder’s Companion Book** is a complete, well-structured learning reference and field guide to the core PyTorch `torch` API.  
It focuses on:

- Deep exploration of `torch.Tensor`, autograd, CUDA, dtype/device handling
- Broadcasting, math ops, FFTs, `torch.linalg`, `torch.special`, and more
- Practical, readable use cases with code-first explanations
- Tools like `torch.utils.data`, `torch.profiler`, and debugging best practices

Whether you’re building deep learning models, exploring low-level tensor logic, or preparing for interviews, this book offers an excellent and visual foundation.

---

## Table of Contents

The book is divided into four parts + appendices:

### Part I: Getting Started
- What is `torch` and Why Does It Matter?
- Installation & Setup
- Tensor Fundamentals

### Part II: torch API Deep Dive
- torch.Tensor: Creation, Ops, Gradients
- Data Types, Devices, and Casting
- Random Sampling & Reproducibility
- Math, Broadcasting, Autograd

### Part III: Specialized Modules
- `torch.linalg` – Linear Algebra
- `torch.nn.functional` – Stateless ops
- `torch.special`, `torch.fft`, `torch.utils`, and low-level storage

### Part IV: Real World
- Using CUDA effectively
- Interfacing with NumPy
- Debugging, Profiling, and Best Practices

### Appendices
- Tensor Shape Cheat Sheet
- PyTorch Idioms & Gotchas
- Full torch API Reference Crosswalk

---

## Getting Started Locally

If you’d like to clone this and run it locally:

```bash
git clone https://github.com/McKlay/pytorch-companion-book.git
cd pytorch-companion-book
pip install mkdocs-material
mkdocs serve
View at: http://127.0.0.1:8000
```
---

Contributing
Contributions, ideas, and corrections are welcome!

Feel free to open:

PRs to add or improve chapters

Issues for broken examples or typos

Discussions for expansion (e.g., torchvision, torch.fx, torch.compile)

---

License  
MIT License © Clay Mark Sarte  
You are free to fork, learn from, and build upon this work — with credit.  
“Shape your tensors, or they will shape your debugging sessions.”  
