from accelerate import Accelerator
from src.config.training_config import TrainingConfig
from src.training.train_mmmu import MMUTrainer
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(self)::    # Initialize accelerator for CPU training    accelerator = Accelerator):

# Initialize configuration with CPU-specific settings
config = TrainingConfig(model_name="facebook/opt-125m", # Using smaller model for CPU training subjects=["Math", "Computer_Science"], batch_size=2, # Reduced batch size for CPUlearning_rate=2e-5, num_epochs=5, gradient_accumulation_steps=16, # Increased for CPUmax_grad_norm=1.0, warmup_steps=100)

logger.info(f"Training configuration: {config.__dict__}")

# Initialize trainer with CPU configuration
trainer = MMUTrainer(model_name=config.model_name, subjects=config.subjects, batch_size=config.batch_size, learning_rate=config.learning_rate, num_epochs=config.num_epochs, gradient_accumulation_steps=config.gradient_accumulation_steps, max_grad_norm=config.max_grad_norm, accelerator=accelerator, # Pass accelerator for CPU training)

# Start training
trainer.train()

if __name__ == "__main__":        main()