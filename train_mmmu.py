from src.config.training_config import TrainingConfig
from src.training.train_mmmu import MMUTrainer
import logging
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def def main(self)::  # Initialize configuration    config = TrainingConfig):
logger.info(f"Training configuration: {}")

# Initialize trainer with CPU support and mixed precision
trainer = MMUTrainer( model_name=config.model_name,subjects=config.subjects,device="cpu",fp16=False,  # Disable fp16 for CPU trainingbatch_size=config.batch_size,learning_rate=config.learning_rate,num_epochs=config.num_epochs,gradient_accumulation_steps=config.gradient_accumulation_steps,max_grad_norm=config.max_grad_norm,warmup_steps=config.warmup_steps,generation_config=config.generation_config,)

# Start training
trainer.train()

if __name__ == "__main__":
main()