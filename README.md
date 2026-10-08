# COL774 Machine Learning Coursework Repository

<p align="center">
  <strong>End-to-end implementations of core machine learning algorithms and experiments for COL774.</strong>
</p>

---

## 📌 Overview

This repository contains assignment-wise implementations, experiments, and reports for **COL774 (Machine Learning)**.  
It covers a broad progression of ML topics—from linear and logistic regression to neural networks, CNNs, Naive Bayes, SVMs, and oblique decision trees.

The project emphasizes:
- algorithm implementation from scratch,
- model training and evaluation workflows,
- assignment-specific experimentation and analysis.

---

## 🗂 Repository Layout

| Directory | Focus Area |
|---|---|
| `A1/` | Linear Regression (including regularization, feature processing, and competitive variants) |
| `A1.2/` | Logistic Regression and feature selection experiments |
| `A2.1/` | Neural Networks (binary and multiclass settings, optimization variants) |
| `A2.2/` | Convolutional Neural Networks (training/testing pipelines and checker scripts) |
| `A3/` | Assignment 3 parts (`A3.1`, `A3.2`, `A3.3`) with train/predict workflows and reports |
| `A4.1/` | Naive Bayes variants and related classification experiments |
| `A5.1/` | Support Vector Machine implementation and learned weights |
| `A5.2/` | Oblique Decision Tree and Logistic Regression based splitting strategies |

---

## 🔍 What You’ll Find

- **Python implementations** for core ML models and pipelines
- **Assignment reports** (`.pdf`) documenting methodology and observations
- **Makefiles / scripts** for reproducible runs in selected assignments
- **Data artifacts and outputs** used for grading, validation, and comparison

---

## 🚀 Quick Start

1. Navigate to the assignment folder you want to run.
2. Install dependencies (if a `requirements.txt` is provided).
3. Use the assignment’s scripts or `makefile` targets to train/test models.

Example:

```bash
cd A2.2
pip install -r requirements.txt
```

---

## 🧠 Academic Scope

This repository is structured for coursework and experimentation.  
Each assignment directory is largely self-contained, with its own scripts, reports, and output artifacts.

---

## 📄 Note

Some assignments reference external datasets or large files that are intentionally not stored in the repository.  
Please follow per-assignment instructions (where available) to set up required data locally.
