from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from pathlib import Path import os
from dataclasses import dataclass, field

from torch.utils.data from tqdm import tqdm import DataLoader
from pathlib import Path import os import logging


"""Module containing specific functionality."""

from torch.utils.data
from tqdm


class Trainer:
    """Class implementing Trainer functionality."""

Base trainer implementation."""Method for __init__.."""Module containing specific functionality."""Method for train_step.."""Module containing specific functionality."""Method for evaluate.."""Module containing specific functionality."""Method for train.."""
        self.step = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Training loop
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training epoch {epoch+1}/{num_epochs}"
            )

            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                self.step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    loss=epoch_loss / num_batches,
                    refresh=False
                )

            # Validation
            val_metrics = self.evaluate()

            if val_metrics and val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                # Save best model checkpoint here if needed

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train loss: {epoch_loss/num_batches:.4f} - "
                f"Val loss: {val_metrics.get('val_loss', 'N/A')}"
            )

        return {
            "train_loss": epoch_loss / num_batches,
            "val_loss": val_metrics.get("val_loss", None)
        }
