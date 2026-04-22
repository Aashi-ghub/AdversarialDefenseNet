from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from utils.logger import MetricLogger


def build_optimizer_and_scheduler(model, config, total_epochs: int):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
        eta_min=config.scheduler.eta_min,
    )
    return optimizer, scheduler


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        checkpoint_dir,
        logger=None,
        metric_logger: MetricLogger | None = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.metric_logger = metric_logger
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, train_loader, val_loader, epochs: int, experiment_name: str, attack_fn=None):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        best_val_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True, attack_fn=attack_fn)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["learning_rate"].append(current_lr)

            row = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "train_accuracy": round(train_metrics["accuracy"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "learning_rate": current_lr,
            }
            if self.metric_logger is not None:
                self.metric_logger.log({"experiment": experiment_name, **row})
            if self.logger is not None:
                self.logger.info(
                    "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.6f",
                    epoch,
                    epochs,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    current_lr,
                )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "history": history,
            }
            self.save_checkpoint(checkpoint, f"{experiment_name}_last.pth")

            if val_metrics["accuracy"] >= best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                self.save_checkpoint(checkpoint, f"{experiment_name}_best.pth")

        if self.metric_logger is not None:
            history_path = self.metric_logger.csv_path.with_name(f"{experiment_name}_history.json")
            self.metric_logger.save_history(history, history_path)

        return history

    def _run_epoch(self, data_loader, training: bool, attack_fn=None):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if training and attack_fn is not None:
                self.model.eval()
                adversarial_images = attack_fn(self.model, images, labels)
                self.model.train()
                inputs = adversarial_images.detach()
            else:
                inputs = images

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / max(1, total),
            "accuracy": correct / max(1, total),
        }

    def evaluate(self, data_loader):
        return self._run_epoch(data_loader, training=False, attack_fn=None)

    def save_checkpoint(self, checkpoint: dict, filename: str) -> Path:
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path


def build_criterion():
    return nn.CrossEntropyLoss()
