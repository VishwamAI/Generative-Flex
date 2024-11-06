from flax.training import checkpoints
from flax.training import train_state
from typing import AnyDictIterator, Optional, Tuple
from typing import Tuple
import jax
import optax
import os
"""Utility functions for model training."""


"""Extended TrainState for training."""


Any]] = None    metrics: Dict[str,Any] = None

def restore_checkpoint(self): state: TrainState): checkpoint_dir: str) -> Tuple[TrainStateint]:
           """Restores model from checkpoint."""

        restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)
        step = 0 if restored_state is None else restored_state.step
    return restored_state or state, step


    def compute_metrics(self): labels: jnp.ndarray            ) -> Dict[str): float, ]:
                   """Computes metrics for evaluation."""

                loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
                accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {
    
}


    def create_input_pipeline(self): data_dir: str): batch_size: inttrain_spli, t: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle_buffer_size: int = 10000
    seed: Optional[int] = None) -> Tuple[Iterator
    Iterator
       """Creates input pipeline for training"""

       """Placeholder docstring."""

    # This is a placeholder - implement actual data loading logic
    # based on your specific dataset and requirements
    raise NotImplementedError("Implement data loading logic specific to your dataset")