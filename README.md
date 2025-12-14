# SMS Spam Detection using DistilBERT

A comprehensive deep learning project for detecting spam messages using transfer learning with DistilBERT. This project includes baseline evaluation, fine-tuning, optimization experiments, and multiple improvement techniques.

## Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Experiments & Results](#-experiments--results)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [Key Insights](#-key-insights)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## Overview

This project implements a state-of-the-art spam detection system using DistilBERT, a distilled version of BERT that maintains 97% of BERT's performance while being 60% faster. The system achieves over 98% accuracy on spam classification tasks through various optimization techniques and transfer learning approaches.

## Features

- **Baseline Model Evaluation**: Pre-trained DistilBERT assessment before fine-tuning
- **Transfer Learning**: Fine-tuning on SMS spam dataset
- **Multiple Optimization Techniques**:
  - Optimizer comparison (AdamW vs RMSProp)
  - Learning rate tuning
  - Data augmentation using synonym replacement
  - Dropout regularization
  - Cross-dataset transfer learning
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Model Persistence**: Saved models for deployment

## Dataset

The project uses the [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv):

- **Total Messages**: 5,572
- **Ham (Non-spam)**: 4,825 (86.6%)
- **Spam**: 747 (13.4%)
- **Split Ratio**: 70% Train / 15% Validation / 15% Test

Additional evaluation on [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) for transfer learning validation.

## Model Architecture

### Base Model
- **Architecture**: DistilBERT-base-uncased
- **Parameters**: ~66M
- **Tokenizer**: DistilBertTokenizerFast
- **Max Sequence Length**: 128 tokens
- **Classification Head**: Linear layer (768 → 2)

### Training Configuration
- **Frozen Layers**: All DistilBERT base layers
- **Trainable Parameters**: Classification head only
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5 (optimized)
- **Batch Size**: 16
- **Epochs**: 3-5 (depending on experiment)

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Dependencies
```bash
pip install transformers datasets scikit-learn torch nlpaug nltk
```

### NLTK Data
```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```



## Experiments & Results

### 1. Baseline Evaluation (Before Fine-tuning)

| Metric | Value |
|--------|-------|
| Accuracy | 81.34% |
| Precision | 21.05% |
| Recall | 14.29% |
| F1-Score | 17.02% |
| Loss | 0.674 |

**Analysis**: The pre-trained model shows poor performance on spam detection, indicating the need for domain-specific fine-tuning.

### 2. Initial Fine-tuning

| Metric | Value |
|--------|-------|
| Accuracy | 98.80% |
| Precision | 95.54% |
| Recall | 95.54% |
| F1-Score | 95.54% |
| Loss | 0.042 |

**Improvement**: +17.46% accuracy, demonstrating the effectiveness of transfer learning.

### 3. Optimizer Comparison

#### AdamW Performance

| Learning Rate | Accuracy | Precision | Recall | F1-Score | Loss |
|--------------|----------|-----------|--------|----------|------|
| 5e-5 | 98.80% | 98.11% | 92.86% | 95.41% | 0.039 |
| 1e-4 | 99.04% | 100.00% | 92.86% | 96.30% | 0.036 |

#### RMSProp Performance

| Learning Rate | Accuracy | Precision | Recall | F1-Score | Loss |
|--------------|----------|-----------|--------|----------|------|
| 5e-5 | 98.80% | 98.11% | 92.86% | 95.41% | 0.038 |
| 1e-4 | 99.04% | 100.00% | 92.86% | 96.30% | 0.034 |

**Best Configuration**: RMSProp with lr=1e-4 (lowest loss: 0.034)

### 4. Learning Rate Fine-tuning

| Configuration | Accuracy | Precision | Recall | F1-Score | Loss |
|--------------|----------|-----------|--------|----------|------|
| AdamW lr=1e-5 | 98.56% | 96.30% | 92.86% | 94.55% | 0.045 |
| AdamW lr=3e-5 | 98.56% | 96.30% | 92.86% | 94.55% | 0.041 |

**Finding**: Learning rate between 5e-5 and 1e-4 provides optimal performance.

### 5. Data Augmentation (Synonym Replacement)

- **Original Training Size**: 3,900 samples
- **Augmented Training Size**: 4,200 samples (+300 augmented)
- **Technique**: WordNet-based synonym augmentation

| Metric | Value |
|--------|-------|
| Accuracy | 99.28% |
| Precision | 99.07% |
| Recall | 95.54% |
| F1-Score | 97.27% |
| Loss | 0.034 |

**Improvement**: +0.48% accuracy over baseline fine-tuning

### 6. Dropout Regularization

- **Dropout Rate**: 0.3
- **Epochs**: 5

| Metric | Value |
|--------|-------|
| Accuracy | 99.40% |
| Precision | 99.08% |
| Recall | 96.43% |
| F1-Score | 97.74% |
| Loss | 0.032 |

**Best Overall Performance**: Highest accuracy and F1-score achieved

### 7. Transfer Learning on New Dataset

Evaluated on UCI SMS Spam Collection:

| Metric | Value |
|--------|-------|
| Accuracy | 98.78% |
| Precision | 96.41% |
| Recall | 94.32% |
| F1-Score | 95.36% |
| Loss | 0.037 |

**Finding**: Model generalizes well to different spam datasets, confirming robust transfer learning.

## Usage
## Performance Metrics

### Overall Best Model (Dropout + 5 Epochs)
```
Accuracy:  99.40%
Precision: 99.08%
Recall:    96.43%
F1-Score:  97.74%
Loss:      0.032
```

### Confusion Matrix Analysis
- **True Negatives**: High accuracy on ham messages
- **True Positives**: Excellent spam detection
- **False Positives**: Minimal (< 1%)
- **False Negatives**: Very low (< 4%)

## Key Insights

1. **Transfer Learning is Effective**: +17.46% accuracy improvement from baseline
2. **Optimizer Choice Matters**: RMSProp slightly outperforms AdamW at higher learning rates
3. **Data Augmentation Helps**: +0.48% accuracy with synonym replacement
4. **Dropout Prevents Overfitting**: Best overall performance with 0.3 dropout rate
5. **Model Generalizes Well**: Strong performance on different datasets (98.78% accuracy)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- **Dataset**: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Base Model**: HuggingFace Transformers - DistilBERT
- **Framework**: PyTorch, Transformers, scikit-learn

