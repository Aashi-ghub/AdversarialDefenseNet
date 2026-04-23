from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from adv_project.attacks.fgsm import fgsm_attack
from adv_project.attacks.pgd import pgd_attack
from adv_project.models.cnn import CustomCNN
from adv_project.models.resnet import build_resnet18
from adv_project.utils.config import CIFAR10_MEAN, CIFAR10_STD


APP_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DISPLAY_SIZE = (320, 320)

MODEL_SPECS = {
    "CNN (Baseline)": {
        "builder": lambda: CustomCNN(num_classes=10),
        "paths": [
            APP_ROOT / "model.pth",
            APP_ROOT / "adv_project" / "checkpoints" / "cnn_standard_best.pth",
        ],
    },
    "ResNet-18": {
        "builder": lambda: build_resnet18(num_classes=10, use_pretrained=False),
        "paths": [
            APP_ROOT / "resnet_model.pth",
            APP_ROOT / "adv_project" / "checkpoints" / "resnet18_standard_best.pth",
        ],
    },
}


CUSTOM_CSS = """
:root {
  --bg: #0f1115;
  --bg-soft: #151821;
  --panel: rgba(26, 26, 26, 0.92);
  --panel-strong: rgba(20, 23, 30, 0.96);
  --panel-muted: rgba(255, 255, 255, 0.03);
  --text: #eef2f7;
  --muted: #aab4c3;
  --accent: #10a37f;
  --accent-alt: #3b82f6;
  --danger: #ef4444;
  --border: rgba(255, 255, 255, 0.08);
  --shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
}

html, body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 28%),
    radial-gradient(circle at top right, rgba(16, 163, 127, 0.14), transparent 22%),
    linear-gradient(180deg, #0d1015 0%, #0b0d12 100%) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  max-width: 1480px !important;
  padding: 30px 22px 40px !important;
}

.hero {
  background: linear-gradient(135deg, rgba(16, 163, 127, 0.13), rgba(59, 130, 246, 0.11));
  border: 1px solid var(--border);
  border-radius: 28px;
  padding: 30px 32px;
  margin-bottom: 20px;
  box-shadow: var(--shadow);
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(16, 163, 127, 0.12);
  border: 1px solid rgba(126, 234, 216, 0.18);
  color: #a7f3d0;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.hero-title {
  margin: 0;
  font-size: 44px;
  line-height: 1.05;
  color: #f7f9fc;
  letter-spacing: -0.03em;
}

.hero-subtitle {
  margin: 12px 0 0;
  max-width: 760px;
  font-size: 18px;
  color: #b4bfcd;
  line-height: 1.6;
}

.glass-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 20px !important;
  box-shadow: 0 18px 44px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(14px);
}

.section-label {
  color: #8fa7c9;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 12px;
}

.gr-button#analyze-btn {
  background: linear-gradient(90deg, var(--accent), var(--accent-alt)) !important;
  color: #f8fafc !important;
  border: none !important;
  border-radius: 16px !important;
  font-weight: 700 !important;
  height: 50px;
  box-shadow: 0 14px 30px rgba(16, 163, 127, 0.22);
  transition: transform 0.2s ease, filter 0.2s ease, box-shadow 0.2s ease;
}

.gr-button#analyze-btn:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}

.gr-button#analyze-btn:active {
  transform: translateY(0);
}

.gr-box, .gr-form, .gr-panel, .gr-accordion, .gr-group {
  border-color: var(--border) !important;
}

textarea, input, select, .gr-dropdown, .gr-textbox, .gr-radio, .gr-slider, .gr-number {
  background: var(--bg-soft) !important;
  color: var(--text) !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
}

.gradio-container label, .gradio-container p, .gradio-container span {
  color: #dbe4ef !important;
}

.image-shell {
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: #11131a;
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}

.image-shell:hover {
  transform: translateY(-2px);
  border-color: rgba(59, 130, 246, 0.24);
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
}

.prediction-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}

.prediction-card {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 18px;
  min-height: 210px;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.prediction-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.18);
}

.prediction-card.flipped {
  border-color: rgba(239, 68, 68, 0.34);
  box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.12), 0 18px 34px rgba(239, 68, 68, 0.09);
}

.card-kicker {
  color: #90a8c8;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 12px;
}

.card-value {
  color: #f7fafc;
  font-size: 28px;
  font-weight: 800;
  line-height: 1.2;
  letter-spacing: -0.02em;
}

.card-confidence {
  margin-top: 8px;
  color: #bfccda;
  font-size: 15px;
}

.confidence-list {
  margin-top: 14px;
  display: grid;
  gap: 10px;
}

.confidence-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
  color: #e3ebf5;
  font-size: 14px;
}

.status-banner {
  padding: 14px 16px;
  border-radius: 18px;
  font-size: 15px;
  font-weight: 700;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
}

.status-banner.stable {
  background: rgba(16, 163, 127, 0.12);
  color: #c9ffef;
  border-color: rgba(16, 163, 127, 0.18);
}

.status-banner.flipped {
  background: rgba(239, 68, 68, 0.14);
  color: #ffd0d0;
  border-color: rgba(239, 68, 68, 0.24);
}

.status-banner.missing {
  background: rgba(59, 130, 246, 0.12);
  color: #dbeafe;
  border-color: rgba(59, 130, 246, 0.2);
}

.footer-note {
  margin-top: 8px;
  color: #8b96a8;
  font-size: 13px;
}

#upload-image .image-frame,
#original-image .image-frame,
#adversarial-image .image-frame,
#heatmap-image .image-frame {
  border-radius: 20px !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  background: #10131a !important;
}
"""


def _find_checkpoint_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _sanitize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict


def load_model(model_label: str, builder, candidate_paths: list[Path]):
    checkpoint_path = _find_checkpoint_path(candidate_paths)
    if checkpoint_path is None:
        return None, None

    model = builder().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = _sanitize_state_dict(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint_path


def load_available_models():
    models = {}
    metadata = {}
    for label, spec in MODEL_SPECS.items():
        model, checkpoint_path = load_model(label, spec["builder"], spec["paths"])
        if model is not None:
            models[label] = model
            metadata[label] = checkpoint_path
    return models, metadata


AVAILABLE_MODELS, MODEL_METADATA = load_available_models()
DEFAULT_MODEL = next(iter(AVAILABLE_MODELS.keys()), "CNN (Baseline)")


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    resize_transform = transforms.Resize((32, 32))
    normalized_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    resized_image = resize_transform(image)
    tensor = normalized_transform(image).unsqueeze(0).to(DEVICE)
    return resized_image, tensor


def denormalize_tensor(image_tensor: torch.Tensor):
    mean = torch.tensor(CIFAR10_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=image_tensor.device).view(1, 3, 1, 1)
    return torch.clamp(image_tensor * std + mean, 0.0, 1.0)


def tensor_to_pil(image_tensor: torch.Tensor):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image = denormalize_tensor(image_tensor)[0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image).resize(DISPLAY_SIZE, Image.Resampling.BICUBIC)


def softmax_bundle(logits: torch.Tensor):
    probabilities = F.softmax(logits, dim=1)[0]
    top_indices = torch.argsort(probabilities, descending=True)[:3]
    top_scores = probabilities[top_indices]
    prediction_index = int(top_indices[0].item())
    return {
        "index": prediction_index,
        "label": CLASS_NAMES[prediction_index],
        "confidence": float(probabilities[prediction_index].item()),
        "topk": [
            (CLASS_NAMES[int(index.item())], float(score.item()))
            for index, score in zip(top_indices.detach().cpu(), top_scores.detach().cpu())
        ],
    }


def apply_attack(model, image_tensor, attack_name: str, epsilon: float, alpha: float, steps: int):
    with torch.no_grad():
        clean_logits = model(image_tensor)
    clean_info = softmax_bundle(clean_logits)
    labels = torch.tensor([clean_info["index"]], device=DEVICE)

    if attack_name == "None":
        adversarial_tensor = image_tensor.clone().detach()
    elif attack_name == "FGSM":
        adversarial_tensor = fgsm_attack(
            model=model,
            images=image_tensor,
            labels=labels,
            epsilon=epsilon,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        )
    else:
        adversarial_tensor = pgd_attack(
            model=model,
            images=image_tensor,
            labels=labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=int(steps),
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
            random_start=True,
        )

    with torch.no_grad():
        adversarial_logits = model(adversarial_tensor)

    adversarial_info = softmax_bundle(adversarial_logits)
    return clean_info, adversarial_info, adversarial_tensor


def create_heatmap(original_image: Image.Image, adversarial_image: Image.Image):
    original_array = np.asarray(original_image).astype(np.float32)
    adversarial_array = np.asarray(adversarial_image).astype(np.float32)
    difference = np.abs(adversarial_array - original_array).mean(axis=2)
    difference = difference / (difference.max() + 1e-8)

    red_channel = np.clip(np.power(difference, 0.75) * 255, 0, 255)
    green_channel = np.clip(np.power(difference, 1.2) * 200 + 30, 0, 255)
    blue_channel = np.clip((1.0 - difference) * 110 + 25, 0, 255)

    heatmap = np.stack([red_channel, green_channel, blue_channel], axis=-1).astype(np.uint8)
    return Image.fromarray(heatmap)


def build_prediction_card(title: str, prediction: dict, flipped: bool = False):
    topk_rows = "".join(
        f"""
        <div class="confidence-row">
            <span>{html.escape(label)}</span>
            <span>{score * 100:.2f}%</span>
        </div>
        """
        for label, score in prediction["topk"]
    )
    flipped_class = " flipped" if flipped else ""
    return f"""
    <div class="prediction-card{flipped_class}">
        <div class="card-kicker">{html.escape(title)}</div>
        <div class="card-value">{html.escape(prediction["label"])}</div>
        <div class="card-confidence">Confidence: {prediction["confidence"] * 100:.2f}%</div>
        <div class="confidence-list">{topk_rows}</div>
    </div>
    """


def build_status_banner(clean_label: str, adversarial_label: str, attack_name: str):
    if attack_name == "None":
        return (
            '<div class="status-banner stable">Clean inference selected. '
            "No adversarial perturbation has been applied.</div>"
        )
    if clean_label != adversarial_label:
        return (
            f'<div class="status-banner flipped">Prediction flipped under {html.escape(attack_name)}. '
            f'The model changed from {html.escape(clean_label)} to {html.escape(adversarial_label)}.</div>'
        )
    return (
        f'<div class="status-banner stable">Prediction stayed stable under {html.escape(attack_name)}. '
        f'The model kept the label {html.escape(clean_label)}.</div>'
    )


def missing_model_banner():
    return (
        '<div class="status-banner missing">Model checkpoint not found. '
        'Place <code>model.pth</code> in the repository root for the CNN, '
        'and optionally <code>resnet_model.pth</code> for ResNet-18.</div>'
    )


def analyze_image(image, model_name, attack_name, epsilon, alpha, steps):
    if image is None:
        message = (
            '<div class="status-banner missing">Upload an image first to run the robustness demo.</div>'
        )
        return None, None, None, build_prediction_card("Clean Prediction", {
            "label": "Waiting for image",
            "confidence": 0.0,
            "topk": [("Upload an image", 1.0)],
        }), build_prediction_card("Adversarial Prediction", {
            "label": "Waiting for image",
            "confidence": 0.0,
            "topk": [("Choose an attack", 1.0)],
        }), message

    if model_name not in AVAILABLE_MODELS:
        placeholder_card = build_prediction_card(
            "Model Status",
            {"label": "Checkpoint missing", "confidence": 0.0, "topk": [("model.pth required", 1.0)]},
        )
        return None, None, None, placeholder_card, placeholder_card, missing_model_banner()

    model = AVAILABLE_MODELS[model_name]
    resized_image, image_tensor = preprocess_image(image)
    clean_info, adversarial_info, adversarial_tensor = apply_attack(
        model=model,
        image_tensor=image_tensor,
        attack_name=attack_name,
        epsilon=float(epsilon),
        alpha=float(alpha),
        steps=int(steps),
    )

    original_display = resized_image.resize(DISPLAY_SIZE, Image.Resampling.BICUBIC)
    adversarial_display = tensor_to_pil(adversarial_tensor)
    heatmap = create_heatmap(original_display, adversarial_display)

    flipped = clean_info["label"] != adversarial_info["label"]
    clean_card = build_prediction_card("Clean Prediction", clean_info, flipped=False)
    adversarial_card = build_prediction_card("Adversarial Prediction", adversarial_info, flipped=flipped)
    status_banner = build_status_banner(clean_info["label"], adversarial_info["label"], attack_name)

    return original_display, adversarial_display, heatmap, clean_card, adversarial_card, status_banner


def toggle_attack_controls(attack_name):
    epsilon_visible = attack_name in {"FGSM", "PGD"}
    pgd_visible = attack_name == "PGD"
    return (
        gr.update(visible=epsilon_visible),
        gr.update(visible=pgd_visible),
        gr.update(visible=pgd_visible),
    )


def build_app():
    initial_status = (
        missing_model_banner()
        if not AVAILABLE_MODELS
        else '<div class="status-banner stable">Ready. Upload an image and explore how the model behaves under attack.</div>'
    )

    with gr.Blocks(css=CUSTOM_CSS, fill_height=True, title="Adversarial Vision Shield") as demo:
        gr.HTML(
            """
            <div class="hero">
                <div class="hero-badge">Premium demo • CPU ready • Hugging Face Spaces</div>
                <h1 class="hero-title">Adversarial Vision Shield</h1>
                <p class="hero-subtitle">Understanding How AI Models Fail Under Attack. Explore clean inference, FGSM, and PGD against CIFAR-10 image classifiers through a product-style interactive dashboard.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["glass-card"]):
                gr.HTML('<div class="section-label">Control Center</div>')
                image_input = gr.Image(
                    label="Drag & Drop an Image",
                    type="pil",
                    elem_id="upload-image",
                    elem_classes=["image-shell"],
                    height=320,
                )
                model_choices = list(AVAILABLE_MODELS.keys()) if AVAILABLE_MODELS else [DEFAULT_MODEL]
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Model",
                )
                attack_dropdown = gr.Radio(
                    choices=["None", "FGSM", "PGD"],
                    value="None",
                    label="Attack Type",
                )
                epsilon_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.2,
                    value=0.05,
                    step=0.005,
                    label="Epsilon",
                    visible=False,
                )
                alpha_slider = gr.Slider(
                    minimum=0.001,
                    maximum=0.05,
                    value=0.01,
                    step=0.001,
                    label="PGD Alpha",
                    visible=False,
                )
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=7,
                    step=1,
                    label="PGD Steps",
                    visible=False,
                )
                analyze_button = gr.Button("Analyze Attack Surface", elem_id="analyze-btn")
                gr.HTML(
                    '<div class="footer-note">Tip: the app attacks the model using its own clean prediction as the source label, which is ideal for interactive failure analysis.</div>'
                )

            with gr.Column(scale=8):
                status_output = gr.HTML(initial_status)

                with gr.Row(equal_height=True):
                    original_output = gr.Image(
                        label="Original Image",
                        type="pil",
                        elem_id="original-image",
                        elem_classes=["glass-card", "image-shell"],
                        height=320,
                    )
                    adversarial_output = gr.Image(
                        label="Adversarial Image",
                        type="pil",
                        elem_id="adversarial-image",
                        elem_classes=["glass-card", "image-shell"],
                        height=320,
                    )

                with gr.Row(equal_height=True):
                    clean_prediction_output = gr.HTML(
                        '<div class="prediction-grid"></div>',
                        elem_classes=["glass-card"],
                    )
                    adversarial_prediction_output = gr.HTML(
                        '<div class="prediction-grid"></div>',
                        elem_classes=["glass-card"],
                    )

                heatmap_output = gr.Image(
                    label="Perturbation Heatmap",
                    type="pil",
                    elem_id="heatmap-image",
                    elem_classes=["glass-card", "image-shell"],
                    height=280,
                )

        attack_dropdown.change(
            fn=toggle_attack_controls,
            inputs=attack_dropdown,
            outputs=[epsilon_slider, alpha_slider, steps_slider],
        )

        analyze_button.click(
            fn=analyze_image,
            inputs=[image_input, model_dropdown, attack_dropdown, epsilon_slider, alpha_slider, steps_slider],
            outputs=[
                original_output,
                adversarial_output,
                heatmap_output,
                clean_prediction_output,
                adversarial_prediction_output,
                status_output,
            ],
            show_progress="full",
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue().launch()
