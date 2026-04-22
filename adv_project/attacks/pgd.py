from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_bounds(mean, std, device):
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)
    lower_bound = (0.0 - mean_tensor) / std_tensor
    upper_bound = (1.0 - mean_tensor) / std_tensor
    return std_tensor, lower_bound, upper_bound


def pgd_attack(
    model,
    images,
    labels,
    epsilon: float,
    alpha: float,
    steps: int,
    mean,
    std,
    random_start: bool = True,
    loss_fn=None,
):
    loss_fn = loss_fn or F.cross_entropy
    std_tensor, lower_bound, upper_bound = _normalized_bounds(mean, std, images.device)

    scaled_epsilon = epsilon / std_tensor
    scaled_alpha = alpha / std_tensor

    if random_start:
        delta = torch.empty_like(images).uniform_(-1.0, 1.0) * scaled_epsilon
        adv_images = images.detach() + delta
        adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)
    else:
        adv_images = images.detach().clone()

    for _ in range(steps):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        gradients = torch.autograd.grad(loss, adv_images, only_inputs=True)[0]

        adv_images = adv_images + scaled_alpha * gradients.sign()
        perturbation = adv_images - images
        perturbation = torch.max(torch.min(perturbation, scaled_epsilon), -scaled_epsilon)
        adv_images = images + perturbation
        adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound).detach()

    return adv_images
