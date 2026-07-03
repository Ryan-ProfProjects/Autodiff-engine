# Autodiff-engine

A vector-valued reverse-mode automatic differentiation (Autodiff) engine built completely from scratch using only **NumPy**. Inspired by Andrej Karpathy's micrograd, this engine scales scalar autodiff mechanics up to multi-dimensional tensor arrays.

![image](https://github.com/user-attachments/assets/0533ecd6-da5d-48a7-8fba-0eef909c6938)

Known Issues: The current .backward() implementation relies on standard recursion rather than a topological sort. Gradients are updated out of order which can lead to mathematically inaccurate accumulations down the graph.

Work-In-Progess

Packages: NumPy, PyTorch (to test)
