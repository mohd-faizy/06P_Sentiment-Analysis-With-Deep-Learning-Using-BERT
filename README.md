# 🔬 Sentiment Analysis with Deep Learning using BERT

<p align='center'>
  <a href="#"><img src='assets/banner.png'></a>
</p>


[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)
[![Platform](https://img.shields.io/badge/Platform-Visual_Studio_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)
[![Maintained](https://img.shields.io/maintenance/yes/2026?style=for-the-badge)](https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)
[![GitHub issues](https://img.shields.io/github/issues/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/issues)
[![Open Source Love](https://img.shields.io/badge/Open_Source-%E2%99%A5-red?style=for-the-badge&logo=open-source-initiative&logoColor=white)](https://opensource.com/resources/what-open-source)
[![Stars](https://img.shields.io/github/stars/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/stargazers)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Repo Size](https://img.shields.io/github/repo-size/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT?style=for-the-badge)](https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)



## 📌 Table of Contents

- [Overview](#overview)
- [What is BERT?](#-what-is-bert)
- [The Transformer Foundation](#-the-transformer-foundation)
- [BERT Architecture Deep Dive](#-bert-architecture-deep-dive)
- [Pre-training Strategies](#-pre-training-strategies)
- [BERT Variants](#-bert-variants)
- [Project Implementation](#-project-implementation)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Key References](#-key-references)
- [Connect](#connect-with-me)

---

## Overview

This project demonstrates **fine-tuning a pre-trained BERT model** for multi-class **sentiment analysis** on Twitter data using **PyTorch** and the **Hugging Face Transformers** library. The model classifies tweets into **6 emotion categories**: `happy`, `not-relevant`, `angry`, `disgust`, `sad`, and `surprise`.

> **Key Takeaway**: Instead of training a language model from scratch (which requires massive compute — BERT-Large was trained on **16 TPUs for 4 days**), we leverage transfer learning by fine-tuning a pre-trained BERT checkpoint, achieving strong performance with minimal task-specific data.

---

## 🤖 What is BERT?

**Bidirectional Encoder Representations from Transformers (BERT)** is a transformer-based language representation model developed by [Jacob Devlin et al. at Google AI Language (2018)](https://arxiv.org/abs/1810.04805). BERT revolutionized NLP by introducing **deep bidirectional pre-training** — unlike previous models (e.g., OpenAI GPT) that read text left-to-right, BERT reads text in **both directions simultaneously**.

<img src='assets/bert.png'>

### Why BERT Matters

| Feature | Traditional Models (Word2Vec, GloVe) | BERT |
|---------|--------------------------------------|------|
| Context | **Context-free** — same vector for "bank" in "river bank" and "bank account" | **Context-aware** — different representations based on surrounding words |
| Directionality | N/A or Unidirectional | **Deeply Bidirectional** — attends to both left and right context in all layers |
| Pre-training | Word-level embeddings only | Full model pre-training on **BooksCorpus (800M words) + English Wikipedia (2,500M words)** |
| Transfer Learning | Limited | Fine-tune with **just one additional output layer** for any downstream task |

### Key Specifications

- **Bidirectional**: BERT is naturally bi-directional, conditioning on both left and right context
- **Generalizable**: Pre-trained BERT can be fine-tuned easily for any downstream NLP task
- **High Performance**: Fine-tuned BERT achieves state-of-the-art on 11 NLP tasks
- **Universal**: Pre-trained on **3.3 billion words** — entire Wikipedia + BooksCorpus

> 🏆 **Recognition**: BERT won the **Best Long Paper Award** at NAACL 2019, and Google adopted BERT for search queries across **70+ languages** by December 2019.

---

## ⚡ The Transformer Foundation

BERT is built on the **Transformer architecture** proposed in the landmark paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). The Transformer replaced recurrent (RNN/LSTM) networks with a **pure attention-based mechanism**, enabling massive parallelization.

### Self-Attention Mechanism

The core of the Transformer is **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$ (queries), $K$ (keys), and $V$ (values) are projections of the input. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large in magnitude.

### Multi-Head Attention

Instead of performing a single attention function, the Transformer uses **multi-head attention** — splitting queries, keys, and values into $h$ heads, applying attention in parallel, and concatenating the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

This allows the model to **jointly attend to information from different representation subspaces** at different positions.

### Why Transformers Beat RNNs

| Property | RNNs/LSTMs | Transformer |
|----------|------------|-------------|
| Sequential Operations | $O(n)$ — must process tokens one by one | $O(1)$ — all positions processed in parallel |
| Maximum Path Length | $O(n)$ — long-range dependencies are hard | $O(1)$ — constant path length between any two positions |
| Parallelization | Limited | **Fully parallelizable** |

<img src='assets/transformer_architecture.jpg'>

---

## 🏗️ BERT Architecture Deep Dive

BERT uses **only the encoder** portion of the Transformer (no decoder needed, since BERT generates representations, not sequences).

### Model Configurations

| Model | Layers (L) | Hidden Size (H) | Attention Heads (A) | Parameters |
|-------|-----------|-----------------|--------------------|-----------:|
| **BERT-Base** | 12 | 768 | 12 | **110M** |
| **BERT-Large** | 24 | 1024 | 16 | **340M** |

> **Note**: Feed-forward filter size is always $4H$ (3072 for Base, 4096 for Large).

### Input Representation

BERT's input embedding is the **sum of three embeddings**:

![Embeddings](assets/bert_embeddings.jpg)

1. **Token Embeddings** — WordPiece tokenization with a 30,000 token vocabulary. Special tokens: `[CLS]` (classification) prepended to every input, and `[SEP]` (separator) between/after sentences.
2. **Segment Embeddings** — Distinguishes Sentence A from Sentence B (for sentence-pair tasks).
3. **Position Embeddings** — Encodes the position of each token in the sequence.

### Architecture Components (as used in this project)

```
BertForSequenceClassification
├── BertEmbeddings         → Input embedding layer (Token + Segment + Position)
├── BertEncoder            → 12 BERT attention layers (for bert-base)
├── BertPooler             → Pools [CLS] token's hidden state
└── Classifier             → Custom linear layer for 6-class sentiment output
```

---

## 📖 Pre-training Strategies

BERT is pre-trained using **two unsupervised tasks** on unlabeled text:

### Task 1: Masked Language Model (MLM)

- **15% of tokens** are randomly selected for prediction
- Of these selected tokens:
  - **80%** are replaced with `[MASK]`
  - **10%** are replaced with a random token
  - **10%** are left unchanged
- The model predicts the original token using **bidirectional context**
- This is inspired by the Cloze task (Taylor, 1953)

> This strategy solves a key limitation: standard language models can only be trained left-to-right, because bidirectional conditioning would let each word "see itself."

### Task 2: Next Sentence Prediction (NSP)

- Given sentence pairs (A, B):
  - **50%** of the time, B is the **actual next sentence** (label: `IsNext`)
  - **50%** of the time, B is a **random sentence** (label: `NotNext`)
- The `[CLS]` token output is used for this binary classification
- The final model achieves **97-98% accuracy** on NSP

> NSP enables BERT to understand **inter-sentence relationships**, critical for tasks like question answering and natural language inference.

### Fine-tuning

Fine-tuning is remarkably efficient compared to pre-training:
- Add **one task-specific output layer** on top of pre-trained BERT
- Fine-tune **all parameters end-to-end** with labeled data
- Typical hyperparameters: learning rate **2e-5 to 5e-5**, batch size **32**, epochs **3-4**
- Can be done in **~1 hour on a single Cloud TPU** or a few hours on a GPU

![BERT Fine-tuning](assets/bert_finetuning.png)

---

## 🔀 BERT Variants

BERT has inspired numerous variants and extensions:

### Performance Variants

| Variant | Key Innovation |
|---------|---------------|
| **RoBERTa** | Trained longer, on more data, with bigger batches, longer sequences; removed NSP; uses dynamic masking |
| **ALBERT** | Parameter reduction techniques; uses Sentence-Order Prediction (SOP) instead of NSP |
| **XLNet** | Uses permutation instead of masking; combines BERT's denoising autoencoding with Transformer-XL's autoregressive modeling |
| **MT-DNN** | BERT + multi-task training on NLU tasks; cross-task data for regularization |
| **SpanBERT** | Masks contiguous spans instead of random tokens |

### Compression Variants

| Variant | Key Innovation |
|---------|---------------|
| **DistilBERT** | 60% faster, 40% smaller while retaining 97% of BERT's performance |
| **TinyBERT** | Knowledge distillation for edge deployment |
| **ALBERT** | Cross-layer parameter sharing |

### Multilingual & Domain-Specific

- **mBERT** — Multilingual BERT (104 languages)
- **CamemBERT** — French
- **AraBERT** — Arabic
- **VisualBERT** — Vision + Language tasks
- **K-BERT** — Knowledge-enhanced BERT
- **BioBERT** — Biomedical text mining
- **SciBERT** — Scientific publications

---

## 🛠️ Project Implementation

### Workflow

The project follows a systematic 10-step workflow:

> **Step 01** — Introduction to Sentiment Analysis and the neural network approach

> **Step 02** — Exploratory Data Analysis: explore dataset distribution, handle class imbalance, and preprocess (remove multi-label & `nocode` entries)

> **Step 03** — Stratified train/validation split (90/10) preserving class distribution

> **Step 04** — Tokenization using `BertTokenizer` (`bert-base-uncased`) — encodes text into `input_ids`, `attention_mask` tensors (max length: 256, padding + truncation)

> **Step 05** — Load `BertForSequenceClassification` with pre-trained weights and a custom 6-class output layer

> **Step 06** — Create `DataLoader` objects with `RandomSampler` (training) and `SequentialSampler` (validation), batch size: 32

> **Step 07** — Configure **AdamW** optimizer (lr=1e-5, eps=1e-8) with linear warmup scheduler

> **Step 08** — Define performance metrics: **weighted F1 score** and **per-class accuracy**

> **Step 09** — Training loop with **gradient clipping** (max norm 1.0) for 10 epochs, supporting both CPU and GPU

> **Step 10** — Load fine-tuned model and evaluate per-class performance

### Why These Choices?

- **`bert-base-uncased`**: Computationally efficient (110M params) while retaining strong performance; uncased works well for informal social media text
- **AdamW**: Applies weight decay before gradient step, better regularization than standard Adam
- **Gradient Clipping**: Prevents exploding gradients in deep transformer networks
- **Linear Warmup Scheduler**: Gradually increases learning rate to prevent early training instability

---

## 📊 Dataset

**[SMILE Twitter Emotion Dataset](https://doi.org/10.6084/m9.figshare.3187909.v2)**

> Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016)

### Dataset Statistics (after preprocessing)

| Category | Count | Proportion |
|----------|------:|:----------:|
| 😊 happy | 1,137 | 78.5% |
| 🤷 not-relevant | 214 | 14.8% |
| 😠 angry | 57 | 3.9% |
| 😲 surprise | 35 | 2.4% |
| 😢 sad | 3 | 0.2% |
| 🤢 disgust | 2 | 0.1% |

> ⚠️ **Class Imbalance**: The dataset exhibits significant class imbalance. The `happy` class dominates with ~78% of samples, while `sad` and `disgust` have <1%. This is addressed during evaluation with stratified splitting and the F1 metric.

### Preprocessing Steps
1. Removed tweets with **multiple emotion labels** (pipe-separated categories)
2. Removed tweets labeled as **`nocode`** (no identifiable emotion)
3. Created label encoding: `{happy: 0, not-relevant: 1, angry: 2, disgust: 3, sad: 4, surprise: 5}`

---

## 📈 Results

### Per-Class Accuracy (Fine-tuned BERT — Epoch 10)

| Class | Accuracy |
|-------|----------|
| happy | **168/171 (98.2%)** |
| not-relevant | 16/32 (50.0%) |
| angry | 0/9 (0.0%) |
| disgust | 0/1 (0.0%) |
| sad | 0/1 (0.0%) |
| surprise | 0/2 (0.0%) |

> **Analysis**: The model shows excellent performance on the majority class (`happy`) but struggles with minority classes due to severe class imbalance. Potential improvements include:
> - **Data augmentation** for underrepresented classes
> - **Oversampling / SMOTE** techniques
> - **Class-weighted loss function** to penalize misclassification of minority classes
> - Using a **larger, more balanced dataset**

### Benchmark Context: BERT on Standard NLP Tasks

For reference, BERT achieved the following state-of-the-art results on major benchmarks:

| Benchmark | BERT-Base | BERT-Large | Previous SOTA |
|-----------|-----------|------------|---------------|
| GLUE (Average) | 79.6 | **82.1** | 75.1 (OpenAI GPT) |
| MNLI (Acc.) | 84.6 | **86.7** | 82.1 (OpenAI GPT) |
| SQuAD v1.1 (F1) | 88.5 | **93.2** | 91.7 (Ensemble) |
| SQuAD v2.0 (F1) | — | **83.1** | 78.0 (Previous best) |
| SST-2 (Acc.) | 93.5 | **94.9** | 93.2 |

---

## 📂 Project Structure

```
06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/
│
├── Sentiment_Analysis_using_BERT.ipynb                        # Main notebook (Colab-ready)
├── data/
│   └── smile-annotations-final.csv                              # SMILE Twitter dataset
├── assets/
│   ├── adamw_algorithm.png                                      # AdamW decoupled weight decay diagram
│   ├── banner.png                                               # Project header banner
│   ├── bert.png                                                 # BERT architecture diagram
│   ├── bert_embeddings.jpg                                      # BERT input embeddings representation
│   ├── bert_finetuning.png                                      # BERT fine-tuning workflow diagram
│   ├── flatten_layer.png                                        # Flatten layer representation
│   ├── multi_head_attention.png                                 # Multi-head attention architecture
│   ├── pytorch_dataloader.png                                   # PyTorch DataLoader architecture diagram
│   ├── transformer_architecture.jpg                             # Transformer encoder-decoder architecture
│   └── transformer_model.png                                    # Transformer model sequence mechanism
├── README.md                                                    # This file
└── LICENSE                                                      # MIT License
```

---

## 🚀 Installation & Usage

### Prerequisites

```bash
pip install torch torchvision
pip install tqdm
pip install transformers
pip install scikit-learn
pip install pandas
```

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT.git
   cd 06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT
   ```

2. **Run on Google Colab** (Recommended for GPU access)
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/blob/master/Sentiment_Analysis_using_BERT.ipynb)

3. **Run Locally**
   - Open the notebook in Jupyter or VS Code
   - Ensure dataset is available in `data/smile-annotations-final.csv`
   - Execute cells sequentially

> ⚡ **GPU Recommendation**: Fine-tuning BERT is compute-intensive. Using a GPU (NVIDIA CUDA) or Google Colab's free GPU runtime is strongly recommended. Training on CPU is possible but significantly slower.

---

## 📚 Key References

### Research Papers

1. **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)** — Devlin, Chang, Lee, Toutanova (2019). The foundational BERT paper introducing masked language modeling and next sentence prediction for bidirectional pre-training.

2. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — Vaswani et al. (2017). The Transformer paper that introduced self-attention as a replacement for recurrence, achieving 28.4 BLEU on WMT English-to-German translation.

### Tutorials & Guides

3. [BERT Explained: A Complete Guide with Theory and Tutorial](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/) — TowardsML
4. [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) — Chris McCormick
5. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
6. [Google AI Blog: Open Sourcing BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

### Video Resources

7. [Transformer Neural Networks - EXPLAINED!](https://youtu.be/TQQlZhbC5ps)
8. [BERT Neural Network - EXPLAINED!](https://youtu.be/xI0HHN5XKDo)
9. [BERT Research Paper Walkthrough](https://youtu.be/-9evrZnBorM)
10. [Illustrated Guide to Transformers](https://youtu.be/4Bdc55j80l8)

### Official Repositories

- [Google Research BERT](https://github.com/google-research/bert)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)


---

## 🔗 Connect with Me

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/F4izy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohd-faizy/)
[![Stack Exchange](https://img.shields.io/badge/Stack_Exchange-1E5397?style=for-the-badge&logo=stack-exchange&logoColor=white)](https://ai.stackexchange.com/users/36737/faizy)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)

</div>

<div align="center">
<strong>⭐ If you found this project helpful, please consider giving it a star!</strong>
</div>