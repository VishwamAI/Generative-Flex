"""Script to fix train_mmmu.py formatting."""


def create_fixed_content():
    """Create properly formatted content for train_mmmu.py."""
    content = '''"""Training script for MMMU dataset using enhanced transformer model.

This module implements the training loop and evaluation logic for the
enhanced transformer model on the MMMU (Massive Multitask Mathematical Understanding)
dataset. It includes logging, checkpointing, and performance monitoring.
"""

import os
import time
import logging
from typing import Dict, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

from src.models.enhanced_transformer import EnhancedTransformer
from src.config.config import ModelConfig
from src.data.mmmu_dataloader import MMMUDataLoader
from src.training.utils.logging import setup_logging


def setup_training(config: ModelConfig):
    """Set up training environment and model.

    Args:
        config: Model configuration object

    Returns:
        Tuple of (model, optimizer, initial_state)
    """
    model = EnhancedTransformer(config=config)
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    initial_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), jnp.ones((1, 1))),
        tx=optimizer
    )
    return model, optimizer, initial_state


def train_step(state, batch, config):
    """Perform single training step.

    Args:
        state: Current training state
        batch: Batch of training data
        config: Model configuration

    Returns:
        Updated state and metrics
    """
    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch["input_ids"],
            batch["attention_mask"]
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["labels"]
        ).mean()
        return loss, {"loss": loss}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def evaluate(state, eval_ds, config):
    """Evaluate model on validation dataset.

    Args:
        state: Current training state
        eval_ds: Validation dataset
        config: Model configuration

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = []
    for batch in eval_ds:
        logits = state.apply_fn(
            state.params,
            batch["input_ids"],
            batch["attention_mask"]
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["labels"]
        ).mean()
        metrics.append({"loss": loss})

    # Average metrics across batches
    avg_metrics = {
        k: jnp.mean([m[k] for m in metrics])
        for k in metrics[0].keys()
    }
    return avg_metrics


def log_metrics(metrics: Dict[str, Any], step: int, prefix: str = ""):
    """Log training metrics to console and file.

    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        prefix: Optional prefix for metric names
    """
    log_str = f"Step {step}"
    for name, value in metrics.items():
        if prefix:
            name = f"{prefix}_{name}"
        log_str += f", {name}: {value:.4f}"
    logging.info(log_str)


def main():
    """Main training function."""
    # Setup
    config = ModelConfig()
    setup_logging()

    # Initialize model and training state
    model, optimizer, state = setup_training(config)

    # Load data
    data_loader = MMMUDataLoader(config)
    train_ds = data_loader.get_train_dataset()
    eval_ds = data_loader.get_eval_dataset()

    # Training loop
    logging.info("Starting training...")
    for step in range(config.max_steps):
        # Training step
        batch = next(train_ds)
        state, metrics = train_step(state, batch, config)

        # Log training metrics
        if step % config.log_every == 0:
            log_metrics(metrics, step, prefix="train")

        # Evaluation
        if step % config.eval_every == 0:
            eval_metrics = evaluate(state, eval_ds, config)
            log_metrics(eval_metrics, step, prefix="eval")

        # Save checkpoint
        if step % config.save_every == 0:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint_{step}")
            state.save(checkpoint_dir)


    logging.info("Training complete!")


if __name__ == "__main__":
    main()
'''
    return content


def main():
    """Main function to fix the file."""
    # Create the fixed content
    content = create_fixed_content()

    # Write to file
    with open("src/training/train_mmmu.py", "w") as f:
        f.write(content)
    print("Fixed train_mmmu.py with proper docstring formatting")


if __name__ == "__main__":
    main()
