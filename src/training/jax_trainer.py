"""JAX-based trainer implementation.."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from src.models import BaseModel
from src.utils.training_utils import TrainingUtils

@dataclass
class JaxTrainerConfig:
    """Configuration for JAX trainer.."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    gradient_clip_norm: float = 1.0
    device: str = "gpu"
    mixed_precision: bool = True
    optimizer_params: Dict = field(default_factory=dict)

class JaxTrainer:
    """JAX-based model trainer.."""

    def __init__(self, model: BaseModel, config: Optional[JaxTrainerConfig] = None):
        """Initialize JAX trainer.

        Args:
            model: Model to train
            config: Optional trainer configuration"""
        self.model = model
        self.config = config or JaxTrainerConfig()
        self.utils = TrainingUtils()

    def train_step(self, state: Dict, batch: Dict) -> Tuple[Dict, float]:
        """Perform single training step.

        Args:
            state: Current training state
            batch: Batch of training data

        Returns:
            Updated state and loss value"""
        def loss_fn(params):
            logits = self.model.apply(params, batch["input_ids"])
            loss = jnp.mean(
                self.utils.compute_loss(logits, batch["labels"])
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state["params"])

        # Clip gradients
        grads = self.utils.clip_gradients(
            grads,
            self.config.gradient_clip_norm
        )

        # Update parameters
        state = self.utils.update_params(
            state,
            grads,
            self.config.learning_rate
        )

        return state, loss

    def train(self, train_data: Dict) -> Dict:
        """Train model on provided data.

        Args:
            train_data: Training dataset

        Returns:
            Training metrics"""
        state = self.utils.init_training_state(
            self.model,
            self.config
        )

        for epoch in range(self.config.num_epochs):
            for batch in self.utils.get_batches(
                train_data,
                self.config.batch_size
            ):
                state, loss = self.train_step(state, batch)

            # Log metrics
            metrics = {
                "loss": loss,
                "epoch": epoch
            }
            self.utils.log_metrics(metrics)

        return metrics
