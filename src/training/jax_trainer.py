"""JAX-based trainer implementation."""
import os
from typing import Dict, Any, Optional, List, Union, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

class JaxTrainer:
    """JAX trainer class for model training."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
    ):
        """Initialize JAX trainer.

        Args:
            model: Flax model to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for learning rate schedule
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps

        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=self._lr_schedule,
            weight_decay=weight_decay,
        )

        # Initialize training state
        self.state = None

    def _lr_schedule(self, step: int) -> float:
        """Learning rate schedule with linear warmup."""
        warmup_factor = jnp.minimum(step / self.warmup_steps, 1.0)
        return self.learning_rate * warmup_factor

    def create_state(self, rng: jnp.ndarray, input_shape: Tuple) -> train_state.TrainState:
        """Create initial training state.

        Args:
            rng: JAX random number generator
            input_shape: Shape of input tensors

        Returns:
            Initial training state
        """
        variables = self.model.init(rng, jnp.ones(input_shape))
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=self.optimizer,
        )
        return self.state

    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Perform single training step.

        Args:
            state: Current training state
            batch: Batch of training data

        Returns:
            Updated state and metrics
        """
        def loss_fn(params):
            outputs = state.apply_fn(
                {"params": params},
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                outputs, batch["labels"]
            ).mean()
            return loss, outputs

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, outputs), grads = grad_fn(state.params)

        # Clip gradients
        grads = optax.clip_by_global_norm(grads, self.max_grad_norm)

        # Update state
        state = state.apply_gradients(grads=grads)

        metrics = {
            "loss": loss,
            "learning_rate": self._lr_schedule(state.step),
        }

        return state, metrics

    def evaluate(
        self,
        state: train_state.TrainState,
        eval_ds: Dict[str, jnp.ndarray],
    ) -> Dict[str, float]:
        """Evaluate model on validation data.

        Args:
            state: Current training state
            eval_ds: Validation dataset

        Returns:
            Evaluation metrics
        """
        outputs = state.apply_fn(
            {"params": state.params},
            eval_ds["input_ids"],
            attention_mask=eval_ds.get("attention_mask"),
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            outputs, eval_ds["labels"]
        ).mean()

        metrics = {
            "eval_loss": loss,
        }
        return metrics
