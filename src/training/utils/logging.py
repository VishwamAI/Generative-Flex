import json
import os
from datetime import datetime
from typing import Dict, Any


class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir,
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        self.metrics_history = []

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics for a training step"""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.metrics_history.append(log_entry)

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        config_file = os.path.join(self.log_dir, "training_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else {}

    def save_summary(self):
        """Save a summary of the training run"""
        summary = {
            "start_time": (
                self.metrics_history[0]["timestamp"]
                if self.metrics_history
                else None
            ),
            "end_time": (
                self.metrics_history[-1]["timestamp"]
                if self.metrics_history
                else None
            ),
            "total_steps": len(self.metrics_history),
            "final_metrics": self.get_latest_metrics(),
        }

        summary_file = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
