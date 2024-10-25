"""Utility functions for model training."""

import os
from typing import Any, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state
from flax.training import checkpoints
import tensorflow as tf


class TrainState(train_state.TrainState):
    """Extended TrainState for training."""

    batch_stats: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = None


def create_train_state(
    rng: jnp.ndarray,
    model: flax.linen.Module,
    input_shape: Tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
) -> TrainState:
    """Creates initial training state."""
    variables = model.init(rng, jnp.ones(input_shape))

    # Create Adam optimizer with weight decay
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables.get("batch_stats"),
        metrics={"loss": 0.0, "accuracy": 0.0},
    )


def save_checkpoint(state: TrainState, checkpoint_dir: str, step: int) -> None:
    """Saves model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state, step=step, keep=3
    )


def restore_checkpoint(
    state: TrainState, checkpoint_dir: str
) -> Tuple[TrainState, int]:
    """Restores model from checkpoint."""
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir, target=state
    )
    step = 0 if restored_state is None else restored_state.step
    return restored_state or state, step


def create_data_iterator(
    dataset: tf.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator:
    """Creates data iterator from tensorflow dataset."""
    if shuffle:
        dataset = dataset.shuffle(10000, seed=seed)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    def iterator():
        for batch in dataset:
            yield jax.tree_map(lambda x: x.numpy(), batch)

    return iterator()


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, float]:
    """Computes metrics for evaluation."""
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return {"loss": loss, "accuracy": accuracy}


def create_learning_rate_scheduler(
    base_learning_rate: float,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 5,
) -> optax.Schedule:
    """Creates learning rate scheduler with warmup and cosine decay."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=base_learning_rate, transition_steps=warmup_steps
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=total_steps - warmup_steps
    )

    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
    )


def create_input_pipeline(
    data_dir: str,
    batch_size: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle_buffer_size: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[Iterator, Iterator, Iterator]:
    """Creates input pipeline for training, validation and testing."""
    # This is a placeholder - implement actual data loading logic
    # based on your specific dataset and requirements
    raise NotImplementedError("Implement data loading logic specific to your dataset")
