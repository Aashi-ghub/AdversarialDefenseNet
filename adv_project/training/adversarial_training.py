from __future__ import annotations

from functools import partial

from attacks.pgd import pgd_attack
from training.trainer import Trainer


class PGDAdversarialTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        checkpoint_dir,
        mean,
        std,
        epsilon: float,
        alpha: float,
        steps: int,
        logger=None,
        metric_logger=None,
        use_amp: bool = True,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
            metric_logger=metric_logger,
            use_amp=use_amp,
        )
        self.attack_fn = partial(
            pgd_attack,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            mean=mean,
            std=std,
            random_start=True,
        )

    def fit(self, train_loader, val_loader, epochs: int, experiment_name: str):
        return super().fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            experiment_name=experiment_name,
            attack_fn=self.attack_fn,
        )


def build_pgd_attack_from_config(config):
    return partial(
        pgd_attack,
        epsilon=config.attack.pgd_epsilon,
        alpha=config.attack.pgd_alpha,
        steps=config.attack.pgd_steps,
        mean=config.data.mean,
        std=config.data.std,
        random_start=True,
    )
