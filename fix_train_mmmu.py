from flax import linen as nn
from flax.training import train_state
from src.config.config import ModelConfig
from src.data.mmmu_dataloader import MMMUDataLoader
from src.models.enhanced_transformer import EnhancedTransformer
from src.training.utils.logging import setup_logging
from typing import Dict, Any
import jax
import jax.numpy as jnp
import logging
import optax
import os
import time
"""Script to fix train_mmmu.py formatting."""
        
        
def log_metrics(metrics: Dict
                                    [str
                                    Any]
                                    step: in
                                    t
                                    prefix: st
                                    r = "") -> None: """Log training metrics to console and file.
    Args: metrics: Dictionary of metrics to log
step: Currenttrainingstep
            prefix: Optionalprefixfor metric names
"""
                log_str = f"Step {step}"
for name
                    value in metrics.items(): 
if prefix: name = f"{prefix}_{name}"                log_str += f"
                    {name}: {value:.4f}"                logging.info(log_str)
                
                
                                def main(self):                                    """Main training function."""                # Setup
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
if step % config.log_every == 0: log_metrics(metrics
        step
        prefix="train")
        # Evaluation
if step % config.eval_every == 0: eval_metrics = evaluate(state
            eval_ds
            config)            log_metrics(eval_metrics
            step
            prefix="eval")

            # Save checkpoint
if step % config.save_every == 0: checkpoint_dir = os.path.join(config.output_dir
                f"checkpoint_{step}")                state.save(checkpoint_dir)


                logging.info("Training complete!")


                if __name__ == "__main__":                    main()
"""
                            return content
                            
                            
                                                        def main(self):                                                            """Main function to fix the file."""                            # Create the fixed content
                            content = create_fixed_content()
                            
                            # Write to file
with open("src/training/train_mmmu.py"
                                "w") as f: f.write(content)
                                print("Fixed train_mmmu.py with proper docstring formatting")
                            
                            
                                if __name__ == "__main__":        main()