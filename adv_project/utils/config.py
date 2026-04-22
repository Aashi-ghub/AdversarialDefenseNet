from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class PathConfig:
    project_root: str = "."
    data_dir: str = "data"
    checkpoints_dir: str = "checkpoints"
    outputs_dir: str = "outputs"
    logs_dir: str = "outputs/logs"
    plots_dir: str = "outputs/plots"
    metrics_dir: str = "outputs/metrics"

    def resolve(self, value: str) -> Path:
        return (Path(self.project_root) / value).resolve()


@dataclass
class DataConfig:
    batch_size: int = 128
    num_workers: int = 2
    validation_split: float = 0.1
    image_size: int = 32
    pin_memory: bool = True
    mean: tuple[float, float, float] = CIFAR10_MEAN
    std: tuple[float, float, float] = CIFAR10_STD


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4


@dataclass
class SchedulerConfig:
    eta_min: float = 1e-5


@dataclass
class AttackConfig:
    fgsm_epsilon: float = 8 / 255
    fgsm_curve_epsilons: list[float] = field(
        default_factory=lambda: [0.0, 2 / 255, 4 / 255, 8 / 255, 12 / 255, 16 / 255]
    )
    pgd_epsilon: float = 8 / 255
    pgd_alpha: float = 2 / 255
    pgd_steps: int = 7


@dataclass
class ExperimentConfig:
    seed: int = 42
    num_classes: int = 10
    train_epochs: int = 5
    adv_train_epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    run_models: tuple[str, ...] = ("cnn", "resnet18")
    adv_models: tuple[str, ...] = ("cnn", "resnet18")


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def create_directories(self) -> None:
        directories = [
            self.paths.data_dir,
            self.paths.checkpoints_dir,
            self.paths.outputs_dir,
            self.paths.logs_dir,
            self.paths.plots_dir,
            self.paths.metrics_dir,
        ]
        for directory in directories:
            self.paths.resolve(directory).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return file_path


def build_config(project_root: str | Path = ".") -> ProjectConfig:
    project_root = str(Path(project_root).resolve())
    config = ProjectConfig(paths=PathConfig(project_root=project_root))
    config.create_directories()
    return config


def get_device(config: ProjectConfig) -> torch.device:
    if config.experiment.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
