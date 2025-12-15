# SMS Spam Detection Using Fine-Tuned BERT

This repository presents a deep learning–based approach for **SMS spam classification** using a fine-tuned **BERT-base-uncased** model.
The project systematically evaluates multiple optimization stages, including optimizer selection, regularization techniques, learning rate strategies, and training enhancements, with performance assessed using standard classification metrics.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset Description](#dataset-description)
* [Model Architecture](#model-architecture)
* [Training Setup](#training-setup)
* [Experimental Methodology](#experimental-methodology)
* [Results and Analysis](#results-and-analysis)

  * [Baseline Performance](#baseline-performance)
  * [Optimizer Comparison](#optimizer-comparison)
  * [Data Augmentation](#data-augmentation)
  * [Regularization Techniques](#regularization-techniques)
  * [Early Stopping](#early-stopping)
  * [Learning Rate Strategies](#learning-rate-strategies)
* [Final Best Configuration](#final-best-configuration)
* [Conclusion](#conclusion)

---

## Project Overview

Spam messages remain a significant challenge in mobile communication systems. Traditional machine learning approaches rely heavily on manual feature engineering, which limits their ability to capture contextual semantics.

In this project, we employ **BERT (Bidirectional Encoder Representations from Transformers)** to leverage contextual language representations for SMS spam detection. The focus of this work is not only on fine-tuning BERT, but also on **analyzing the impact of different optimization strategies** on classification performance.

---

## Dataset Description

The experiments are conducted on an SMS spam dataset containing two classes:

* **Ham (Non-Spam)**
* **Spam**

Due to the inherent class imbalance in spam datasets, evaluation emphasizes **F1-score and Recall**, as accurate spam detection is more critical than raw accuracy.

The dataset is split into:

* Training set
* Validation set
* Test set

using stratified sampling to preserve class distribution.

---

## Model Architecture

* **Base Model**: BERT-base-uncased
* **Transformer Layers**: 12
* **Hidden Size**: 768
* **Tokenizer**: BertTokenizerFast
* **Classification Head**: Linear layer with softmax activation (binary classification)

During training:

* The BERT encoder is used as a feature extractor.
* The classification head is trained for the downstream spam detection task.

---

## Training Setup

* **Batch Size**: Fixed across experiments
* **Epochs**: Fixed across experiments
* **Evaluation Metrics**:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Loss

To ensure fair comparison, **only one factor is modified per experiment**, while all other hyperparameters remain constant.

---

## Experimental Methodology

The experiments are organized into **progressive optimization stages**:

1. Baseline fine-tuning
2. Optimizer comparison
3. Data augmentation
4. Regularization techniques
5. Early stopping
6. Learning rate strategies

Each stage is evaluated independently, and the best-performing configuration is selected based on F1-score.

---

## Results and Analysis

### Baseline Performance

| Model                      | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Loss |
| -------------------------- | ------------ | ------------- | ---------- | ------------ | ---- |
| Fine-Tuned BERT (Baseline) | 74.22        | 69.01         | 87.50      | 77.17        | 0.65 |

This baseline establishes the reference performance before applying optimization techniques.

---

### Optimizer Comparison

| Optimizer   | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Loss     |
| ----------- | ------------ | ------------- | ---------- | ------------ | -------- |
| AdamW       | 83.11        | 78.91         | 90.18      | 84.17        | 0.60     |
| **RMSprop** | **84.89**    | **81.97**     | 89.29      | **85.47**    | **0.58** |
| SGD         | 77.33        | 72.26         | 88.39      | 79.52        | 0.63     |

**RMSprop** achieved the highest F1-score and lowest loss, making it the best optimizer in this setup.

---

### Data Augmentation

| Model             | Accuracy | F1-score |
| ----------------- | -------- | -------- |
| Baseline Training | 0.7422   | 0.7717   |
| Random Deletion   | 0.7422   | 0.7717   |

Data augmentation using random deletion **did not improve performance**, indicating that the dataset already provided sufficient variability for this task.

---

### Regularization Techniques (Dropout)

| Model                 | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Loss     |
| --------------------- | ------------ | ------------- | ---------- | ------------ | -------- |
| No Dropout            | 74.22        | 69.01         | 87.50      | 77.17        | 0.65     |
| **Dropout (p = 0.3)** | **83.11**    | **78.91**     | **90.18**  | **84.17**    | **0.60** |

Applying dropout significantly reduced overfitting and improved generalization.

---

### Early Stopping

| Model                   | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Loss     |
| ----------------------- | ------------ | ------------- | ---------- | ------------ | -------- |
| Baseline                | 74.22        | 69.01         | 87.50      | 77.17        | 0.65     |
| **With Early Stopping** | **83.11**    | **78.91**     | **90.18**  | **84.17**    | **0.60** |

Early stopping improved performance while reducing unnecessary training epochs.

---

### Learning Rate Strategies

| Learning Rate Strategy   | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Loss     |
| ------------------------ | ------------ | ------------- | ---------- | ------------ | -------- |
| Fixed LR (Baseline)      | 74.22        | 69.01         | 87.50      | 77.17        | 0.65     |
| **Reduce LR on Plateau** | **83.11**    | **78.91**     | **90.18**  | **84.17**    | **0.60** |
| Linear Warmup Scheduler  | 82.67        | 78.29         | 90.18      | 83.82        | 0.60     |

Reducing the learning rate on performance plateaus yielded the most stable convergence.

---

## Final Best Configuration

Based on all experimental stages, the optimal setup is:

* **Model**: BERT-base-uncased
* **Optimizer**: RMSprop
* **Regularization**: Dropout (p = 0.3)
* **Early Stopping**: Enabled
* **Learning Rate Strategy**: Reduce LR on Plateau


---

## Conclusion

This study demonstrates that fine-tuning BERT for SMS spam detection can be significantly enhanced through careful optimization. While data augmentation did not improve results in this case, **regularization techniques and adaptive learning rate strategies proved critical** for improving generalization. The findings highlight the importance of systematic experimentation when applying transformer-based models to real-world text classification tasks.

---
