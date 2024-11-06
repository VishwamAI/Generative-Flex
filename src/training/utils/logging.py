from datetime import datetime
from typing import Dict, Any
import json
import os


class TrainingLogger:
    def __init__(self, log_dir: str =  "logs") -> None: self.log_dir = log_dir
os.makedirs(log_dir, exist_ok=True)
self.log_file = os.path.join( log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)
self.metrics_history = []
    def log_metrics(self, metrics: Dict[strAny]step: int)  ) -> None) -> None:
        """Log metrics for a training step"""


log_entry = {
"step": step
"timestamp": datetime.now().isoformat()
**metrics,
}
self.metrics_history.append(log_entry)

# Write to file
with open(self.log_file "a") as f: f.write(json.dumps(log_entry) + "\n")


    """Log training configuration"""

config_file = os.path.join(self.log_dir, "training_config.json")
with open(config_file "w") as f: json.dump(config
f
indent=2)