from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def build_attack_fn(
    attack_name: str,
    epsilon: float,
    mean,
    std,
    alpha: float | None = None,
    steps: int = 7,
):
    attack_name = attack_name.lower()
    if attack_name == "fgsm":
        return partial(fgsm_attack, epsilon=epsilon, mean=mean, std=std)
    if attack_name == "pgd":
        attack_alpha = alpha if alpha is not None else max(epsilon / 4, 1 / 255)
        return partial(
            pgd_attack,
            epsilon=epsilon,
            alpha=attack_alpha,
            steps=steps,
            mean=mean,
            std=std,
            random_start=True,
        )
    raise ValueError("attack_name must be either 'fgsm' or 'pgd'.")


def predict_probabilities(model, images):
    logits = model(images)
    probabilities = F.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    return logits, probabilities, predictions, confidences


def evaluate_clean(model, data_loader, device, criterion=None):
    criterion = criterion or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if was_training:
        model.train()

    return {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total)}


def evaluate_under_attack(model, data_loader, device, attack_fn, criterion=None):
    criterion = criterion or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adversarial_images = attack_fn(model, images, labels)

        with torch.no_grad():
            outputs = model(adversarial_images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if was_training:
        model.train()

    return {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total)}


def robustness_suite(model, data_loader, device, config, criterion=None):
    criterion = criterion or nn.CrossEntropyLoss()

    clean_metrics = evaluate_clean(model, data_loader, device, criterion)
    fgsm_metrics = evaluate_under_attack(
        model,
        data_loader,
        device,
        build_attack_fn(
            attack_name="fgsm",
            epsilon=config.attack.fgsm_epsilon,
            mean=config.data.mean,
            std=config.data.std,
        ),
        criterion,
    )
    pgd_metrics = evaluate_under_attack(
        model,
        data_loader,
        device,
        build_attack_fn(
            attack_name="pgd",
            epsilon=config.attack.pgd_epsilon,
            alpha=config.attack.pgd_alpha,
            steps=config.attack.pgd_steps,
            mean=config.data.mean,
            std=config.data.std,
        ),
        criterion,
    )

    return {
        "clean_accuracy": clean_metrics["accuracy"],
        "fgsm_accuracy": fgsm_metrics["accuracy"],
        "pgd_accuracy": pgd_metrics["accuracy"],
        "clean_loss": clean_metrics["loss"],
        "fgsm_loss": fgsm_metrics["loss"],
        "pgd_loss": pgd_metrics["loss"],
    }


def accuracy_vs_epsilon(
    model,
    data_loader,
    device,
    attack_name: str,
    epsilons,
    mean,
    std,
    alpha: float | None = None,
    steps: int = 7,
):
    accuracies = []
    for epsilon in epsilons:
        attack_fn = build_attack_fn(
            attack_name=attack_name,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            mean=mean,
            std=std,
        )

        metrics = evaluate_under_attack(model, data_loader, device, attack_fn)
        accuracies.append(metrics["accuracy"])

    return {"epsilons": list(epsilons), "accuracies": accuracies}
