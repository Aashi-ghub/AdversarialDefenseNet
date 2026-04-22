from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def denormalize_batch(images, mean, std):
    mean_tensor = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return torch.clamp(images * std_tensor + mean_tensor, 0.0, 1.0)


def plot_training_curves(histories: dict, output_path: str | Path, title: str = "Training Curves"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for experiment_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{experiment_name} train")
        axes[0].plot(epochs, history["val_loss"], linestyle="--", label=f"{experiment_name} val")
        axes[1].plot(epochs, history["train_accuracy"], label=f"{experiment_name} train")
        axes[1].plot(epochs, history["val_accuracy"], linestyle="--", label=f"{experiment_name} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_robustness_curves(curves: dict, output_path: str | Path, title: str = "Accuracy vs Epsilon"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, curve in curves.items():
        x_values = [epsilon * 255 for epsilon in curve["epsilons"]]
        y_values = [accuracy * 100 for accuracy in curve["accuracies"]]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Epsilon (pixel scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def visualize_adversarial_examples(
    model,
    data_loader,
    device,
    attack_fn,
    class_names,
    mean,
    std,
    output_path: str | Path,
    num_images: int = 6,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)
    adversarial_images = attack_fn(model, images, labels)

    with torch.no_grad():
        original_predictions = model(images).argmax(dim=1)
        adversarial_predictions = model(adversarial_images).argmax(dim=1)

    original_images = denormalize_batch(images, mean, std).cpu()
    perturbed_images = denormalize_batch(adversarial_images, mean, std).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))
    if num_images == 1:
        axes = axes.reshape(2, 1)

    for index in range(num_images):
        axes[0, index].imshow(original_images[index].permute(1, 2, 0))
        original_label = class_names[int(labels[index].item())]
        original_prediction = class_names[int(original_predictions[index].item())]
        adversarial_prediction = class_names[int(adversarial_predictions[index].item())]
        axes[0, index].set_title(
            f"Orig: {original_label}\nPred: {original_prediction}"
        )
        axes[0, index].axis("off")

        axes[1, index].imshow(perturbed_images[index].permute(1, 2, 0))
        axes[1, index].set_title(f"Adv Pred: {adversarial_prediction}")
        axes[1, index].axis("off")

    fig.suptitle("Original vs Adversarial Examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if was_training:
        model.train()

    return fig
