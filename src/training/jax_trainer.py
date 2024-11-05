import os
import jax
from typing import Any, Dict, Optional
import optax
from flax.training import train_state
import logging
from pathlib import Path

"""JAX/Flax training infrastructure for Generative-Flex."""


class TrainerState(train_state.TrainState):
    """Custom train state with loss scaling for mixed precision training."""

    loss_scale: Optional[jnp.ndarray] = None


class FlaxTrainer:
    """Advanced trainer implementation using JAX/Flax."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize training state
        self.setup_training_state()

    def setup_training_state(self):
        """Setup training state with optimizer and learning rate schedule."""
        # Create learning rate schedule
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config["training"]["learning_rate"],
            transition_steps=self.config["training"]["warmup_steps"],
        )

        decay_fn = optax.cosine_decay_schedule(
            init_value=self.config["training"]["learning_rate"],
            decay_steps=self.config["training"]["num_epochs"]
            * self.config["training"]["steps_per_epoch"],
        )

        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[self.config["training"]["warmup_steps"]],
        )

        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config["training"]["max_grad_norm"]),
            optax.adamw(
                learning_rate=schedule_fn,
                weight_decay=self.config["training"]["weight_decay"],
            ),
        )

        # Initialize state
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, self.config["model"]["max_seq_length"]))
        variables = self.model.init(rng, dummy_input)

        self.state = TrainerState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=optimizer,
            _loss_scale=(
                jnp.array(2.0**15)
                if self.config["training"].get("fp16", False)
                else None
            ),
        )

    @staticmethod
    def compute_loss(logits, labels):
        """Compute cross entropy loss."""
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()

    def train_step(self, state, batch):
        """Single training step with gradient updates."""

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["input_ids"])
            loss = self.compute_loss(logits, batch["labels"])
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)

        # Update state
        state = state.apply_gradients(grads=grads)

        return state, loss

    def train(
        self,
        train_dataset,
        num_epochs: int,
        eval_dataset=None,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        log_steps: int = 100,
    ):
        """Training loop with evaluation."""
        train_step_jit = jax.jit(self.train_step)

        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            num_steps = 0

            for batch_idx, batch in enumerate(train_dataset):
                self.state, loss = train_step_jit(self.state, batch)
                epoch_loss += loss
                num_steps += 1

                # Logging
                if batch_idx % log_steps == 0:
                    avg_loss = epoch_loss / num_steps
                    logging.info(
                        f"Epoch: {epoch}, Step: {batch_idx}, " f"Loss: {avg_loss:.4f}"
                    )

                # Evaluation
                if eval_dataset is not None and batch_idx % eval_steps == 0:
                    eval_loss = self.evaluate(eval_dataset)
                    logging.info(f"Eval Loss: {eval_loss:.4f}")

                # Save checkpoint
                if batch_idx % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{epoch}-{batch_idx}")

            # End of epoch
            avg_epoch_loss = epoch_loss / num_steps
            logging.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}")
            self.save_checkpoint(f"epoch-{epoch}")

    def evaluate(self, eval_dataset):
        """Evaluation loop."""
        total_loss = 0
        num_steps = 0

        for batch in eval_dataset:
            logits = self.state.apply_fn(
                {"params": self.state.params}, batch["input_ids"]
            )
            loss = self.compute_loss(logits, batch["labels"])
            total_loss += loss
            num_steps += 1

        return total_loss / num_steps

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters
        with open(checkpoint_dir / "model.msgpack", "wb") as f:
            f.write(flax.serialization.to_bytes(self.state))

        # Save config
        with open(checkpoint_dir / "config.msgpack", "wb") as f:
            f.write(flax.serialization.to_bytes(self.config))

        logging.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(path)

        # Load model parameters
        with open(checkpoint_dir / "model.msgpack", "rb") as f:
            self.state = flax.serialization.from_bytes(self.state, f.read())

        # Load config
        with open(checkpoint_dir / "config.msgpack", "rb") as f:
            self.config = flax.serialization.from_bytes(self.config, f.read())

        logging.info(f"Checkpoint loaded from {checkpoint_dir}")
