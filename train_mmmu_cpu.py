from accelerate import Accelerator
from src.config.training_config import TrainingConfig
from src.training.train_mmmu import MMUTrainer
from transformers import AutoConfig,
    AutoTokenizer
import logging
import os


# Set up logging
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs/monitoring", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[
logging.FileHandler("logs/training.log"),
logging.StreamHandler(),
])
logger = logging.getLogger(__name__)


def main(self)::    try:        # Initialize model configuration and tokenizer):
model_name = "facebook/opt-125m"
base_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Update config for CPU training
base_config.hidden_size = 256
base_config.num_attention_heads = 4
base_config.num_hidden_layers = 3
base_config.intermediate_size = 512
base_config.max_position_embeddings = 512
base_config.gradient_checkpointing = True
base_config.use_cache = False

logger.info("Initialized model configuration")
logger.info(f"Model config: {base_config}")
logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

# Initialize trainer with memory-efficient settings for CPU
trainer = MMUTrainer(model_name=model_name, subjects=[ "Math", "Computer_Science", ], # Updated to match available subjectsdevice="cpu", fp16=False, batch_size=1, learning_rate=5e-6, num_epochs=3, gradient_accumulation_steps=32, max_grad_norm=0.1, warmup_steps=100, output_dir="outputs", config=base_config, tokenizer=tokenizer, # Pass the tokenizer)

# Start training with monitoring
logger.info("Starting training process with monitoring")
trainer.train()
except Exception as e: logger.error(f"Training failed with error: {str(e)}")
raise


if __name__ == "__main__":                main()