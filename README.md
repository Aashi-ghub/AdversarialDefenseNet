# Adversarial Robustness in Deep Learning for Image Classification

This repository contains a complete PyTorch project for studying adversarial robustness on CIFAR-10 in Google Colab.

## Highlights

- Custom CNN baseline and ResNet-18 classifier
- Clean training with checkpointing, metric logging, and learning-rate scheduling
- FGSM and PGD adversarial attacks
- PGD-based adversarial training
- Robustness evaluation on clean, FGSM, and PGD settings
- Training curves, robustness curves, and adversarial image visualizations
- Colab notebook that runs the full workflow step by step

## Structure

```text
adv_project/
├── models/
├── training/
├── attacks/
├── evaluation/
├── utils/
├── checkpoints/
├── outputs/
└── notebooks/
```

## Main Notebook

- `adv_project/notebooks/adversarial_robustness_colab.ipynb`

## Notes

- The notebook is designed for Google Colab with GPU enabled.
- CIFAR-10 is downloaded automatically through `torchvision`.
- Generated checkpoints, logs, metrics, and plots are saved under `adv_project/checkpoints` and `adv_project/outputs`.
