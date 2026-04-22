from __future__ import annotations

from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from evaluation.robustness import build_attack_fn
from evaluation.visualization import tensor_to_uint8_image


def _build_preprocess_transform(config):
    return transforms.Compose(
        [
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )


def _topk_dict(probabilities, class_names, k: int = 3):
    top_values, top_indices = torch.topk(probabilities, k=min(k, len(class_names)))
    return {
        class_names[int(index.item())]: float(value.item())
        for value, index in zip(top_values.detach().cpu(), top_indices.detach().cpu())
    }


def _render_heatmap_image(original_tensor, adversarial_tensor, config):
    clean_image = tensor_to_uint8_image(original_tensor, config.data.mean, config.data.std).astype(np.float32) / 255.0
    adversarial_image = (
        tensor_to_uint8_image(adversarial_tensor, config.data.mean, config.data.std).astype(np.float32) / 255.0
    )
    perturbation = np.abs(adversarial_image - clean_image).mean(axis=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(perturbation, cmap="inferno")
    ax.set_title("Perturbation Heatmap")
    ax.axis("off")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    heatmap_image = Image.open(buffer).convert("RGB")
    heatmap_image.load()
    return heatmap_image


def build_gradio_interface(model_registry: dict, class_names, config, device):
    if not model_registry:
        raise ValueError("model_registry must contain at least one trained model.")

    available_models = {label: model for label, model in model_registry.items()}
    preprocess = _build_preprocess_transform(config)

    def run_demo(image, model_choice, attack_choice, epsilon_pixels):
        if image is None:
            return (
                "Please upload an image to start the demo.",
                "Please upload an image to start the demo.",
                None,
                None,
                None,
                None,
            )

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        model = available_models[model_choice]
        model.eval()

        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clean_logits = model(input_tensor)
            clean_probabilities = F.softmax(clean_logits, dim=1)[0]
            clean_prediction = int(clean_probabilities.argmax().item())
            clean_confidence = float(clean_probabilities[clean_prediction].item())

        epsilon = float(epsilon_pixels) / 255.0
        attack_fn = build_attack_fn(
            attack_name=attack_choice.lower(),
            epsilon=epsilon,
            alpha=max(epsilon / 4, 1 / 255),
            steps=config.attack.pgd_steps,
            mean=config.data.mean,
            std=config.data.std,
        )

        labels = torch.tensor([clean_prediction], device=device)
        adversarial_tensor = attack_fn(model, input_tensor, labels)

        with torch.no_grad():
            adversarial_logits = model(adversarial_tensor)
            adversarial_probabilities = F.softmax(adversarial_logits, dim=1)[0]
            adversarial_prediction = int(adversarial_probabilities.argmax().item())
            adversarial_confidence = float(adversarial_probabilities[adversarial_prediction].item())

        adversarial_image = tensor_to_uint8_image(adversarial_tensor[0], config.data.mean, config.data.std)
        heatmap_image = _render_heatmap_image(input_tensor[0], adversarial_tensor[0], config)

        clean_text = (
            f"Clean prediction: {class_names[clean_prediction]} | confidence: {clean_confidence * 100:.2f}%"
        )
        adversarial_text = (
            f"Adversarial prediction: {class_names[adversarial_prediction]} | confidence: {adversarial_confidence * 100:.2f}%"
        )

        return (
            clean_text,
            adversarial_text,
            adversarial_image,
            heatmap_image,
            _topk_dict(clean_probabilities, class_names),
            _topk_dict(adversarial_probabilities, class_names),
        )

    interface = gr.Interface(
        fn=run_demo,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Radio(list(available_models.keys()), value=list(available_models.keys())[0], label="Model"),
            gr.Radio(["FGSM", "PGD"], value="FGSM", label="Attack"),
            gr.Slider(0, 16, value=8, step=1, label="Epsilon (pixel scale)"),
        ],
        outputs=[
            gr.Textbox(label="Clean Prediction"),
            gr.Textbox(label="Adversarial Prediction"),
            gr.Image(type="numpy", label="Adversarial Image"),
            gr.Image(type="pil", label="Perturbation Heatmap"),
            gr.Label(num_top_classes=3, label="Clean Prediction Confidence"),
            gr.Label(num_top_classes=3, label="Adversarial Prediction Confidence"),
        ],
        title="Adversarial Robustness Demo for Image Classification",
        description=(
            "Upload an image, choose a model and an attack, then inspect how adversarial perturbations change "
            "the prediction. `Adv-CNN (PGD Defense)` is the defended CNN trained on PGD adversarial examples."
        ),
    )
    return interface
