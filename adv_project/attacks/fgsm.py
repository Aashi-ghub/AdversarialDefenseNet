from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_bounds(mean, std, device):
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)
    lower_bound = (0.0 - mean_tensor) / std_tensor
    upper_bound = (1.0 - mean_tensor) / std_tensor
    return mean_tensor, std_tensor, lower_bound, upper_bound


def fgsm_attack(
    model,
    images,
    labels,
    epsilon: float,
    mean,
    std,
    loss_fn=None,
):
    loss_fn = loss_fn or F.cross_entropy
    _, std_tensor, lower_bound, upper_bound = _normalized_bounds(mean, std, images.device)

    scaled_epsilon = epsilon / std_tensor
    adv_images = images.detach().clone().requires_grad_(True)

    outputs = model(adv_images)
    loss = loss_fn(outputs, labels)
    gradients = torch.autograd.grad(loss, adv_images, only_inputs=True)[0]

    adv_images = adv_images + scaled_epsilon * gradients.sign()
    adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)
    return adv_images.detach()
