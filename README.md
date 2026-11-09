# DNN Assignment 2
## RNN, LSTM, GRU and Transformer Networks

**Student Name:** Umesha M S  
**Student ID:** 2026AJ05949

GitHub Repository:  
https://github.com/2026aj05949/DNN_Assignment_2

---

# Project Overview

This assignment implements and evaluates Recurrent Neural Networks (RNNs) and Transformer architectures for sequence modeling tasks.

The project is divided into two sections:

## Problem 1: Recurrent Neural Networks

- Vanilla RNN
- LSTM
- GRU
- Human Activity Recognition (HAR)
- Adding Problem Benchmark
- Gradient Analysis
- Hidden-Size Analysis
- Forget Gate Bias Ablation

## Problem 2: Transformers

- Scaled Dot-Product Attention
- Multi-Head Attention
- Transformer Encoder
- Positional Encoding
- Character-Level Language Modeling
- Attention Visualization
- Head Ablation
- Context-Length Scaling Analysis

---

# Dataset Download

The notebooks automatically download all required datasets during execution.

Datasets used:

- UCI Human Activity Recognition (HAR) Dataset
- Tiny Shakespeare Dataset

No manual dataset download is required.

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

# Environment

- Python 3.10+
- PyTorch 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Seaborn
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Problem 1: RNN, LSTM and GRU

## Custom Recurrent Cells

The following recurrent cells were implemented from scratch:

### MyRNNCell

```text
h_t = tanh(Wih x_t + Whh h_(t-1) + b)
```

### MyLSTMCell

Components:

- Input Gate
- Forget Gate
- Output Gate
- Memory Cell

### MyGRUCell

Components:

- Reset Gate
- Update Gate
- Candidate Hidden State

All implementations were verified against the equivalent PyTorch implementations using unit tests.

---

## Human Activity Recognition (HAR)

### Dataset

UCI HAR Dataset

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

### Models Evaluated

- RNN
- LSTM
- GRU

### Best Result

| Model | Accuracy | Macro F1 |
|---------|---------|---------|
| GRU | 90.26% | 0.9018 |

---

## Adding Problem Benchmark

Sequence lengths evaluated:

```text
T = 50
T = 100
T = 200
T = 400
```

Models compared:

- RNN
- LSTM
- GRU

Metric:

```text
Mean Squared Error (MSE)
```

Observations:

- GRU achieved the lowest MSE for shorter sequences.
- Long-range dependency learning becomes more difficult as sequence length increases.
- Performance degrades significantly at T=400.

---

## Gradient Analysis

Gradient norms were analyzed across timesteps for:

- RNN
- LSTM
- GRU

Results show:

- Vanilla RNN suffers from vanishing gradients.
- LSTM and GRU maintain stronger gradient flow through time.

---

## Hidden-Size Sweep

Hidden dimensions evaluated:

```text
16
32
64
128
```

Results indicate:

- Accuracy improves with larger hidden sizes.
- GRU consistently outperforms LSTM across hidden dimensions.
- Performance gains diminish beyond hidden size 64.

---

## Student-Specific Ablation

Student ID:

```text
2026AJ05949
```

Last Digit:

```text
9
```

Required Experiment:

```text
Forget Gate Bias = 0
vs
Forget Gate Bias = 1
```

Sequence Length:

```text
T = 200
```

Result:

A positive forget-gate bias significantly improves memory retention and reduces test MSE.

---

# Problem 2: Transformer

## Scaled Dot-Product Attention

Implemented from scratch:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

Verified against:

```python
torch.nn.functional.scaled_dot_product_attention
```

---

## Multi-Head Attention

Implemented:

- Query Projection (WQ)
- Key Projection (WK)
- Value Projection (WV)
- Output Projection (WO)

Configuration:

```text
d_model = 128
Heads = 4
```

---

## Positional Encoding

Implemented:

### Sinusoidal Positional Encoding

Compared against:

- No Positional Encoding
- Learned Positional Encoding

Result:

Sinusoidal positional encoding achieved the best overall performance.

---

## Transformer Encoder

The Transformer encoder contains:

- Multi-Head Attention
- Residual Connections
- Layer Normalization
- Feed Forward Network
- Dropout

---

# Character-Level Language Model

Dataset:

```text
Tiny Shakespeare
```

Architecture:

```text
Context Length = 128
Embedding Dimension = 128
Attention Heads = 4
Transformer Layers = 4
```

---

## Language Model Results

Validation Loss:

```text
1.894 nats
```

Bits Per Character:

```text
2.7325
```

Generated samples are stored in:

```text
problem2/generated_samples.txt
```

---

## Temperature Study

Temperatures evaluated:

```text
0.2
0.8
1.5
```

Observations:

- 0.2 → More deterministic and repetitive text
- 0.8 → Best balance between coherence and diversity
- 1.5 → Highest diversity but lower readability

---

# Figures Generated

## Problem 1

- class_balance.png
- har_confusion_matrix.png
- gradient_norm.png
- hidden_sweep.png
- adding_mse_vs_T.png
- adding_curves_T200.png
- ablation_forget_bias.png

## Problem 2

- attention_heatmap_head1.png
- attention_heatmap_head2.png
- positional_encoding_ablation.png
- heads_ablation.png
- transformer_learning_curve.png
- context_time.png
- context_memory.png

---

# Results

All experiment outputs are stored in:

```text
results.csv
```

Metrics include:

- Accuracy
- Macro F1 Score
- Test MSE
- Parameter Count
- Training Time
- Validation Loss
- Bits Per Character

---

# Running the Notebooks

### Problem 1

```bash
jupyter notebook problem1/problem1.ipynb
```

### Problem 2

```bash
jupyter notebook problem2/problem2.ipynb
```

---

# Submission Contents

- README.md
- report.pdf
- requirements.txt
- results.csv
- problem1/problem1.ipynb
- problem2/problem2.ipynb
- problem2/generated_samples.txt
- All generated figures

---

# Author

**Umesha M S**  
**Student ID:** 2026AJ05949

Deep Neural Networks Assignment 2