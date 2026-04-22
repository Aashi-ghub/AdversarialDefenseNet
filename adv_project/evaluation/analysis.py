from __future__ import annotations

from pathlib import Path

import pandas as pd


DISPLAY_NAME_MAP = {
    "cnn_standard": "Custom CNN (Standard)",
    "resnet18_standard": "ResNet-18 (Standard)",
    "cnn_adv_pgd": "Custom CNN (PGD Adversarial Training)",
    "resnet18_adv_pgd": "ResNet-18 (PGD Adversarial Training)",
}


def infer_experiment_metadata(experiment_name: str) -> dict:
    if experiment_name.startswith("resnet18"):
        model_name = "ResNet-18"
    else:
        model_name = "Custom CNN"

    if "adv" in experiment_name:
        training_regime = "PGD Adversarial Training"
    else:
        training_regime = "Standard Training"

    return {
        "experiment": experiment_name,
        "model_name": model_name,
        "training_regime": training_regime,
        "display_name": DISPLAY_NAME_MAP.get(experiment_name, experiment_name.replace("_", " ").title()),
    }


def build_results_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    records = []
    for experiment_name, metrics in results.items():
        metadata = infer_experiment_metadata(experiment_name)
        record = {
            **metadata,
            **metrics,
        }
        record["clean_accuracy_pct"] = metrics["clean_accuracy"] * 100
        record["fgsm_accuracy_pct"] = metrics["fgsm_accuracy"] * 100
        record["pgd_accuracy_pct"] = metrics["pgd_accuracy"] * 100
        record["fgsm_drop_pct"] = (metrics["clean_accuracy"] - metrics["fgsm_accuracy"]) * 100
        record["pgd_drop_pct"] = (metrics["clean_accuracy"] - metrics["pgd_accuracy"]) * 100
        records.append(record)

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    preferred_order = [
        "cnn_standard",
        "resnet18_standard",
        "cnn_adv_pgd",
        "resnet18_adv_pgd",
    ]
    frame["sort_order"] = frame["experiment"].apply(
        lambda value: preferred_order.index(value) if value in preferred_order else len(preferred_order)
    )
    frame = frame.sort_values(["sort_order", "display_name"]).drop(columns=["sort_order"]).reset_index(drop=True)
    return frame


def _format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_model_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    comparison_df = results_df[
        (results_df["training_regime"] == "Standard Training")
        & (results_df["model_name"].isin(["Custom CNN", "ResNet-18"]))
    ][["model_name", "clean_accuracy", "fgsm_accuracy", "pgd_accuracy"]].copy()
    comparison_df.columns = ["Model", "Clean Accuracy", "FGSM Accuracy", "PGD Accuracy"]

    for column in ["Clean Accuracy", "FGSM Accuracy", "PGD Accuracy"]:
        comparison_df[column] = comparison_df[column].apply(_format_accuracy)
    return comparison_df.reset_index(drop=True)


def build_defense_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    defense_df = results_df[results_df["model_name"] == "Custom CNN"][
        ["training_regime", "clean_accuracy", "fgsm_accuracy", "pgd_accuracy"]
    ].copy()
    defense_df.columns = ["CNN Variant", "Clean Accuracy", "FGSM Accuracy", "PGD Accuracy"]
    for column in ["Clean Accuracy", "FGSM Accuracy", "PGD Accuracy"]:
        defense_df[column] = defense_df[column].apply(_format_accuracy)
    return defense_df.reset_index(drop=True)


def generate_analysis_report(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "No evaluation results were available to summarize."

    lines = ["Research Insights"]

    average_fgsm = results_df["fgsm_accuracy_pct"].mean()
    average_pgd = results_df["pgd_accuracy_pct"].mean()
    pgd_gap = average_fgsm - average_pgd
    lines.append(
        f"1. PGD is stronger than FGSM in this study: mean FGSM accuracy is {average_fgsm:.2f}% "
        f"while mean PGD accuracy is {average_pgd:.2f}%, a further drop of {pgd_gap:.2f} points. "
        "This happens because PGD repeatedly refines the perturbation inside the epsilon ball instead of taking a single step."
    )

    standard_rows = results_df[results_df["training_regime"] == "Standard Training"]
    cnn_standard = standard_rows[standard_rows["model_name"] == "Custom CNN"]
    resnet_standard = standard_rows[standard_rows["model_name"] == "ResNet-18"]
    if not cnn_standard.empty and not resnet_standard.empty:
        cnn_row = cnn_standard.iloc[0]
        resnet_row = resnet_standard.iloc[0]
        clean_gap = resnet_row["clean_accuracy_pct"] - cnn_row["clean_accuracy_pct"]
        pgd_gap_models = resnet_row["pgd_accuracy_pct"] - cnn_row["pgd_accuracy_pct"]
        lines.append(
            f"2. Depth changes behavior: the standard ResNet-18 differs from the standard CNN by {clean_gap:.2f} clean-accuracy points "
            f"and {pgd_gap_models:.2f} PGD-accuracy points. Deeper residual features often improve representation quality, "
            "but robustness still depends on how smooth and stable the learned decision boundary becomes under attack."
        )

    adv_cnn = results_df[
        (results_df["model_name"] == "Custom CNN")
        & (results_df["training_regime"] == "PGD Adversarial Training")
    ]
    if not cnn_standard.empty and not adv_cnn.empty:
        clean_cnn = cnn_standard.iloc[0]
        robust_cnn = adv_cnn.iloc[0]
        fgsm_gain = robust_cnn["fgsm_accuracy_pct"] - clean_cnn["fgsm_accuracy_pct"]
        pgd_gain = robust_cnn["pgd_accuracy_pct"] - clean_cnn["pgd_accuracy_pct"]
        clean_tradeoff = robust_cnn["clean_accuracy_pct"] - clean_cnn["clean_accuracy_pct"]
        lines.append(
            f"3. PGD adversarial training improves CNN robustness: FGSM accuracy changes by {fgsm_gain:.2f} points and PGD accuracy changes by {pgd_gain:.2f} points "
            f"relative to the standard CNN, while clean accuracy changes by {clean_tradeoff:.2f} points. "
            "This illustrates the central attack-versus-defense tradeoff: robustness usually improves because the network is trained on hard worst-case perturbations."
        )

    strongest_clean = results_df.loc[results_df["clean_accuracy_pct"].idxmax(), "display_name"]
    strongest_pgd = results_df.loc[results_df["pgd_accuracy_pct"].idxmax(), "display_name"]
    lines.append(
        f"4. The best clean model in this run is {strongest_clean}, while the best PGD-robust model is {strongest_pgd}. "
        "For a viva or demo, this makes it easy to discuss why accuracy leadership and robustness leadership are not always the same."
    )

    return "\n\n".join(lines)


def save_analysis_report(report: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
