from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str | Path, experiment_name: str) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{name}.{experiment_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / f"{experiment_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class MetricLogger:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None

    def log(self, metrics: dict) -> None:
        row = {"timestamp": datetime.utcnow().isoformat(), **metrics}
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_history(self, history: dict, json_path: str | Path) -> Path:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return json_path
