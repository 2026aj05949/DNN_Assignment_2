# DNN_Assignment_2# DNN Assignment 2
## Recurrent Neural Networks and Transformers

**Student ID:** 2026AJ05949

GitHub Repository:
https://github.com/2026aj05949/DNN_Assignment_2

---

# Project Overview

This project implements and evaluates recurrent neural networks and Transformer architectures for sequence modeling tasks.

The assignment consists of two parts:

## Problem 1
Recurrent Neural Networks

- Vanilla RNN
- LSTM
- GRU
- Human Activity Recognition (HAR)
- Adding Problem Benchmark
- Gradient Analysis
- Hidden Size Analysis
- Forget Gate Bias Ablation

## Problem 2
Transformers

- Scaled Dot-Product Attention
- Multi-Head Attention
- Transformer Encoder
- Positional Encoding
- Character-Level Language Model
- Attention Visualization
- Head Ablation
- Context-Length Scaling Analysis

---

# Repository Structure

```text
DNN_Assignment_2/
│
├── README.md
├── report.pdf
├── requirements.txt
├── results.csv
│
├── problem1/
│   ├── problem1.ipynb
│   └── figures/
│       ├── class_balance.png
│       ├── gradient_norm.png
│       ├── har_confusion_matrix.png
│       ├── hidden_sweep.png
│       ├── adding_mse_vs_T.png
│       ├── adding_curves_T200.png
│       └── ablation_forget_bias.png
│
└── problem2/
    ├── problem2.ipynb
    ├── generated_samples.txt
    └── figures/
        ├── attention_heatmap_head1.png
        ├── attention_heatmap_head2.png
        ├── positional_encoding_ablation.png
        ├── heads_ablation.png
        ├── transformer_learning_curve.png
        ├── context_time.png
        └── context_memory.png
```

---

# Datasets

## UCI Human Activity Recognition (HAR)

Used for:

- RNN Classification
- LSTM Classification
- GRU Classification

Dataset Statistics:

```text
Training Samples : 7352
Testing Samples  : 2947
Sequence Length  : 128
Features         : 9
Classes          : 6
```

Activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

---

## Tiny Shakespeare

Used for:

- Character-Level Language Modeling

The Transformer learns next-character prediction and generates Shakespeare-style text.

---

# Problem 1: RNN, LSTM and GRU

## Implemented Components

### MyRNNCell

```text
h_t = tanh(Wih x_t + Whh h_(t-1) + b)
```

### MyLSTMCell

Implements:

- Input Gate
- Forget Gate
- Output Gate
- Cell State

### MyGRUCell

Implements:

- Reset Gate
- Update Gate
- Candidate Hidden State

All custom cells were verified against PyTorch implementations using unit tests.

---

# HAR Classification Results

## Best Model

| Model | Best Accuracy | Best Macro-F1 |
|---------|-------------|-------------|
| GRU | 90.26% | 0.9018 |

Results obtained across multiple random seeds and logged in `results.csv`.

---

# Adding Problem Benchmark

Sequence Lengths:

```text
T = 50
T = 100
T = 200
T = 400
```

Models Compared:

- RNN
- LSTM
- GRU

Evaluation Metric:

```text
Mean Squared Error (MSE)
```

Observations:

- GRU achieved the lowest error for shorter sequences.
- Performance degradation was observed as sequence length increased.
- Long-range dependency learning becomes increasingly difficult at T=400.

---

# Additional Analysis

## Gradient Norm Analysis

Compared gradient propagation across:

- RNN