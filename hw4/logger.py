import json
from datetime import datetime
from typing import Any, Dict


class ExperimentLogger:
    """Логгер экспериментов"""

    def __init__(self, log_file="experiments.json"):
        self.log_file = log_file

    def log_experiment(self, experiment_data: Dict[str, Any]):
        """Логирование эксперимента"""
        experiment_data["timestamp"] = datetime.now().isoformat()

        try:
            with open(self.log_file, "r") as f:
                experiments = json.load(f)
        except FileNotFoundError:
            experiments = []

        experiments.append(experiment_data)

        with open(self.log_file, "w") as f:
            json.dump(experiments, f, indent=2)

        print(f"📝 Эксперимент записан в {self.log_file}")
