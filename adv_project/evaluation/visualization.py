from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_batch(images, mean, std):
    mean_tensor = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return torch.clamp(images * std_tensor + mean_tensor, 0.0, 1.0)


def tensor_to_numpy_image(image_tensor, mean, std):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image = denormalize_batch(image_tensor, mean, std)[0]
    return image.permute(1, 2, 0).detach().cpu().numpy()


def tensor_to_uint8_image(image_tensor, mean, std):
    image = tensor_to_numpy_image(image_tensor, mean, std)
    return (image * 255).clip(0, 255).astype(np.uint8)


def perturbation_heatmap(image_tensor, adversarial_tensor, mean, std):
    clean_image = tensor_to_numpy_image(image_tensor, mean, std)
    adversarial_image = tensor_to_numpy_image(adversarial_tensor, mean, std)
    perturbation = np.abs(adversarial_image - clean_image).mean(axis=2)
    return perturbation


def _save_figure(fig, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return fig


def _collect_attack_triplets(
    model,
    data_loader,
    device,
    fgsm_fn,
    pgd_fn,
    num_images: int = 4,
    require_clean_correct: bool = True,
):
    was_training = model.training
    model.eval()

    selected_images = []
    selected_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        if require_clean_correct:
            selection_mask = clean_predictions.eq(labels)
        else:
            selection_mask = torch.ones_like(labels, dtype=torch.bool)

        if selection_mask.any():
            selected_images.append(images[selection_mask].detach())
            selected_labels.append(labels[selection_mask].detach())

        total_selected = sum(batch.size(0) for batch in selected_labels)
        if total_selected >= num_images:
            break

    if selected_images:
        images = torch.cat(selected_images, dim=0)[:num_images]
        labels = torch.cat(selected_labels, dim=0)[:num_images]
    else:
        images, labels = next(iter(data_loader))
        images = images[:num_images].to(device)
        labels = labels[:num_images].to(device)

    fgsm_images = fgsm_fn(model, images, labels)
    pgd_images = pgd_fn(model, images, labels)

    with torch.no_grad():
        clean_predictions = model(images).argmax(dim=1)
        fgsm_predictions = model(fgsm_images).argmax(dim=1)
        pgd_predictions = model(pgd_images).argmax(dim=1)

    if was_training:
        model.train()

    return {
        "images": images.detach(),
        "labels": labels.detach(),
        "fgsm_images": fgsm_images.detach(),
        "pgd_images": pgd_images.detach(),
        "clean_predictions": clean_predictions.detach(),
        "fgsm_predictions": fgsm_predictions.detach(),
        "pgd_predictions": pgd_predictions.detach(),
    }


def plot_training_curves(histories: dict, output_path: str | Path, title: str = "Training Curves"):
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
    return _save_figure(fig, output_path)


def plot_robustness_curves(curves: dict, output_path: str | Path, title: str = "Accuracy vs Epsilon"):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, curve in curves.items():
        x_values = [epsilon * 255 for epsilon in curve["epsilons"]]
        y_values = [accuracy * 100 for accuracy in curve["accuracies"]]
        ax.plot(x_values, y_values, marker="o", linewidth=2.2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Epsilon (pixel scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_attack_comparison_panel(
    model,
    data_loader,
    device,
    fgsm_fn,
    pgd_fn,
    class_names,
    mean,
    std,
    output_path: str | Path,
    num_images: int = 4,
):
    bundle = _collect_attack_triplets(
        model=model,
        data_loader=data_loader,
        device=device,
        fgsm_fn=fgsm_fn,
        pgd_fn=pgd_fn,
        num_images=num_images,
        require_clean_correct=True,
    )

    fig, axes = plt.subplots(3, num_images, figsize=(4 * num_images, 9))
    if num_images == 1:
        axes = axes.reshape(3, 1)

    image_sets = [
        ("Original", bundle["images"], bundle["clean_predictions"]),
        ("FGSM", bundle["fgsm_images"], bundle["fgsm_predictions"]),
        ("PGD", bundle["pgd_images"], bundle["pgd_predictions"]),
    ]

    for row_index, (row_name, row_images, row_predictions) in enumerate(image_sets):
        for column_index in range(num_images):
            image = tensor_to_numpy_image(row_images[column_index], mean, std)
            label = class_names[int(bundle["labels"][column_index].item())]
            prediction = class_names[int(row_predictions[column_index].item())]
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].set_title(f"{row_name}\nTrue: {label} | Pred: {prediction}")
            axes[row_index, column_index].axis("off")

    fig.suptitle("Image Comparison Panel: Original vs FGSM vs PGD", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_perturbation_heatmaps(
    model,
    data_loader,
    device,
    fgsm_fn,
    pgd_fn,
    mean,
    std,
    output_path: str | Path,
    num_images: int = 4,
):
    bundle = _collect_attack_triplets(
        model=model,
        data_loader=data_loader,
        device=device,
        fgsm_fn=fgsm_fn,
        pgd_fn=pgd_fn,
        num_images=num_images,
        require_clean_correct=True,
    )

    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 6))
    if num_images == 1:
        axes = axes.reshape(2, 1)

    for column_index in range(num_images):
        fgsm_heatmap = perturbation_heatmap(
            bundle["images"][column_index],
            bundle["fgsm_images"][column_index],
            mean,
            std,
        )
        pgd_heatmap = perturbation_heatmap(
            bundle["images"][column_index],
            bundle["pgd_images"][column_index],
            mean,
            std,
        )

        axes[0, column_index].imshow(fgsm_heatmap, cmap="inferno")
        axes[0, column_index].set_title("FGSM Perturbation")
        axes[0, column_index].axis("off")

        axes[1, column_index].imshow(pgd_heatmap, cmap="inferno")
        axes[1, column_index].set_title("PGD Perturbation")
        axes[1, column_index].axis("off")

    fig.suptitle("Perturbation Heatmaps", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_misclassification_examples(
    model,
    data_loader,
    device,
    fgsm_fn,
    pgd_fn,
    class_names,
    mean,
    std,
    output_path: str | Path,
    num_images: int = 4,
):
    was_training = model.training
    model.eval()

    selected = []
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        fgsm_images = fgsm_fn(model, images, labels)
        pgd_images = pgd_fn(model, images, labels)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)
            fgsm_predictions = model(fgsm_images).argmax(dim=1)
            pgd_predictions = model(pgd_images).argmax(dim=1)

        change_mask = clean_predictions.eq(labels) & (
            fgsm_predictions.ne(clean_predictions) | pgd_predictions.ne(clean_predictions)
        )

        indices = torch.where(change_mask)[0].tolist()
        for index in indices:
            selected.append(
                {
                    "image": images[index].detach(),
                    "label": labels[index].detach(),
                    "clean_prediction": clean_predictions[index].detach(),
                    "fgsm_image": fgsm_images[index].detach(),
                    "fgsm_prediction": fgsm_predictions[index].detach(),
                    "pgd_image": pgd_images[index].detach(),
                    "pgd_prediction": pgd_predictions[index].detach(),
                }
            )
            if len(selected) >= num_images:
                break
        if len(selected) >= num_images:
            break

    if not selected:
        if was_training:
            model.train()
        return plot_attack_comparison_panel(
            model=model,
            data_loader=data_loader,
            device=device,
            fgsm_fn=fgsm_fn,
            pgd_fn=pgd_fn,
            class_names=class_names,
            mean=mean,
            std=std,
            output_path=output_path,
            num_images=num_images,
        )

    fig, axes = plt.subplots(3, len(selected), figsize=(4 * len(selected), 9))
    if len(selected) == 1:
        axes = axes.reshape(3, 1)

    for column_index, sample in enumerate(selected):
        true_label = class_names[int(sample["label"].item())]
        clean_prediction = class_names[int(sample["clean_prediction"].item())]
        fgsm_prediction = class_names[int(sample["fgsm_prediction"].item())]
        pgd_prediction = class_names[int(sample["pgd_prediction"].item())]

        axes[0, column_index].imshow(tensor_to_numpy_image(sample["image"], mean, std))
        axes[0, column_index].set_title(f"Original\nTrue: {true_label} | Pred: {clean_prediction}")
        axes[0, column_index].axis("off")

        axes[1, column_index].imshow(tensor_to_numpy_image(sample["fgsm_image"], mean, std))
        axes[1, column_index].set_title(f"FGSM Changed To\n{fgsm_prediction}")
        axes[1, column_index].axis("off")

        axes[2, column_index].imshow(tensor_to_numpy_image(sample["pgd_image"], mean, std))
        axes[2, column_index].set_title(f"PGD Changed To\n{pgd_prediction}")
        axes[2, column_index].axis("off")

    fig.suptitle("Misclassification Examples Under Adversarial Attacks", fontsize=15)
    fig.tight_layout()

    if was_training:
        model.train()

    return _save_figure(fig, output_path)


def plot_model_comparison_graph(results_df, output_path: str | Path, title: str = "CNN vs ResNet Comparison"):
    subset = results_df[
        (results_df["training_regime"] == "Standard Training")
        & (results_df["model_name"].isin(["Custom CNN", "ResNet-18"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = np.arange(len(subset))
    width = 0.24

    ax.bar(x_positions - width, subset["clean_accuracy_pct"], width=width, label="Clean")
    ax.bar(x_positions, subset["fgsm_accuracy_pct"], width=width, label="FGSM")
    ax.bar(x_positions + width, subset["pgd_accuracy_pct"], width=width, label="PGD")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(subset["model_name"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_clean_vs_adversarial_bar_chart(
    results_df,
    output_path: str | Path,
    title: str = "Clean vs Adversarial Accuracy",
):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(results_df))
    width = 0.24

    ax.bar(x_positions - width, results_df["clean_accuracy_pct"], width=width, label="Clean")
    ax.bar(x_positions, results_df["fgsm_accuracy_pct"], width=width, label="FGSM")
    ax.bar(x_positions + width, results_df["pgd_accuracy_pct"], width=width, label="PGD")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(results_df["display_name"], rotation=18, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_defense_comparison_chart(
    results_df,
    output_path: str | Path,
    title: str = "Standard CNN vs Adversarially Trained CNN",
):
    subset = results_df[results_df["model_name"] == "Custom CNN"].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = np.arange(len(subset))
    width = 0.24

    ax.bar(x_positions - width, subset["clean_accuracy_pct"], width=width, label="Clean")
    ax.bar(x_positions, subset["fgsm_accuracy_pct"], width=width, label="FGSM")
    ax.bar(x_positions + width, subset["pgd_accuracy_pct"], width=width, label="PGD")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(subset["display_name"], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_path)


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
    was_training = model.training
    model.eval()

    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)
    adversarial_images = attack_fn(model, images, labels)

    with torch.no_grad():
        original_predictions = model(images).argmax(dim=1)
        adversarial_predictions = model(adversarial_images).argmax(dim=1)

    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))
    if num_images == 1:
        axes = axes.reshape(2, 1)

    for index in range(num_images):
        axes[0, index].imshow(tensor_to_numpy_image(images[index], mean, std))
        original_label = class_names[int(labels[index].item())]
        original_prediction = class_names[int(original_predictions[index].item())]
        adversarial_prediction = class_names[int(adversarial_predictions[index].item())]
        axes[0, index].set_title(f"Orig: {original_label}\nPred: {original_prediction}")
        axes[0, index].axis("off")

        axes[1, index].imshow(tensor_to_numpy_image(adversarial_images[index], mean, std))
        axes[1, index].set_title(f"Adv Pred: {adversarial_prediction}")
        axes[1, index].axis("off")

    fig.suptitle("Original vs Adversarial Examples")
    fig.tight_layout()

    if was_training:
        model.train()

    return _save_figure(fig, output_path)
