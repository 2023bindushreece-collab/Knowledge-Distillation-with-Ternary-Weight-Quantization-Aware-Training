# Knowledge Distillation with Ternary-Weight Quantization-Aware Training

## Project Overview

This project implements Knowledge Distillation with Ternary-Weight Quantization-Aware Training on CIFAR-10.

The main goal is to train a compact ResNet18 student model using a stronger ResNet34 teacher while constraining convolutional and fully-connected weights to ternary values:

Wq ∈ {-alpha, 0, +alpha}

## Dataset

CIFAR-10:

- Training: 45,000 images
- Validation: 5,000 images
- Test: 10,000 images

Random seed: 42

## Models

### ResNet34 Teacher

- Best validation accuracy: 96.02%
- Test accuracy: 95.05%

### FP32 ResNet18 Baseline

- Best validation accuracy: 95.64%
- Test accuracy: 95.10%

### Ternary ResNet18 Student

The ternary student uses:

- Temperature T = 4
- KD weight = 0.7
- CE weight = 0.3
- QAT learning rate = 0.001
- Straight-Through Estimator
- Per-layer alpha
- Threshold Delta = 0.7 * mean(abs(W))

Final ternary results:

- Best validation accuracy: 95.56%
- Test accuracy: 94.65%
- Sparsity: 47.56%
- Compression ratio: 15.80x
- Compressed checkpoint size: 2.77 MB

## External Image Testing

The trained ternary ResNet18 was tested on five external real-world images from CIFAR-10 classes:

- airplane
- cat
- dog
- horse
- truck

The model correctly classified all 5 sample images.

External sample accuracy:

5 / 5 = 100%

This is only a small external demonstration. The official CIFAR-10 test accuracy remains 94.65%.

## Repository Files

- `teacher-atdl.ipynb`  
  ResNet34 teacher training

- `student-atdl.ipynb`  
  FP32 baseline, ternary quantization, KD, QAT, compression and ablation

- `test-and-graphs.ipynb`  
  External image testing and ternary weight visualizations

- `requirements.txt`  
  Required Python packages

## Main Results

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| ResNet34 Teacher | 96.02% | 95.05% |
| FP32 ResNet18 | 95.64% | 95.10% |
| Ternary ResNet18 + KD + QAT | 95.56% | 94.65% |

## Compression

Ternary weights are represented using:

- -alpha
- 0
- +alpha

The ternary values are stored using 2-bit packed codes.

Final compression ratio:

15.80x

Compressed checkpoint size:

2.77 MB

## Reproducibility

Random seed: 42

The same CIFAR-10 training and validation split is used for the teacher, baseline and ternary student.
