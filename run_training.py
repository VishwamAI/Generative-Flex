from accelerate import Accelerator
from datasets import load_dataset
from src.data.mmmu_loader import create_mmmu_dataloaders
from src.training.train_mmmu import MMUTrainer
from transformers import AutoTokenizer, AutoConfig
import logging
import os
import torch
"""Script to run MMMU model training with mathematical reasoning focus."""
        
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])
        logger = logging.getLogger(__name__)
        
        
                def initialize_mmmu_dataset(self, subjects, cache_dir="./data/cache"):                    """Initialize and cache MMMU dataset."""        logger.info(f"Initializing MMMU dataset for subjects: {subjects}")
        try: forsubjectin, subjects: forsplitin ["dev", "validation", "test"]:
    logger.info(f"Loading {subject} - {split} split...")
    _ = load_dataset("MMMU/MMMU", subject, split=split, cache_dir=cache_dir)
    logger.info("Successfully initialized all dataset splits")
    return True
    except Exception as e: logger.error(f"Error initializing dataset: {e}")
        raise


def main(self): """Main training function."""try:    # Set up configuration
    model_name = "facebook/opt-125m"  # Smaller model for local training
    subjects = ["Math"]  # Focus only on Math for initial training
    batch_size = 1  # Minimal batch size for memory efficiency
    gradient_accumulation_steps = 16  # Increased for effective batch size of 16
    learning_rate = 1e-5  # Reduced learning rate for stability
    num_epochs = 5  # Reduced epochs for initial testing
    max_length = 256  # Reduced sequence length for memory efficiency
    output_dir = "outputs"
    cache_dir = "./data/cache"

    # Create output and cache directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize accelerator with basic settings
    accelerator = Accelerator(cpu=True, # Force CPU usage initially
    mixed_precision=None, # Disable mixed precision for CPU
    gradient_accumulation_steps=gradient_accumulation_steps)
    logger.info("Initialized Accelerator for training")

    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Max sequence length: {max_length}")

    # Initialize MMMU dataset
    initialize_mmmu_dataset(subjects, cache_dir)

    # Initialize tokenizer and model configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    # Enhanced configuration for mathematical reasoning
    model_config.num_choices = 4  # A, B, C, D options
    model_config.max_position_embeddings = max_length
    model_config.hidden_size = 256  # Reduced for memory efficiency
    model_config.intermediate_size = 1024  # Reduced intermediate size
    model_config.num_attention_heads = 4  # Reduced number of heads
    model_config.num_hidden_layers = 3  # Reduced number of layers
    model_config.num_experts = 4  # Reduced number of experts
    model_config.expert_dim = (
    model_config.hidden_size
    )  # Match expert dimension to hidden size
    model_config.use_flash_attention = False  # Disable flash attention for CPU
    model_config.dropout = 0.1  # Standard dropout rate
    model_config.load_in_8bit = False  # Keep full precision for accuracy
    model_config.use_cache = False  # Disable KV cache to save memory
    model_config.gradient_checkpointing = True  # Enable gradient checkpointing
    model_config.tie_word_embeddings = True  # Enable weight tying for efficiency
    model_config.use_memory_efficient_attention = (
    True  # Enable memory efficient attention
    )
    model_config.attention_probs_dropout_prob = 0.1  # Standard attention dropout
    model_config.hidden_dropout_prob = 0.1  # Standard hidden dropout
    model_config.use_reentrant = (
    True  # Enable reentrant for better memory efficiency
    )
    model_config.image_input_size = 112  # Reduced image size for memory efficiency

    # Initialize trainer with enhanced settings and accelerator
    trainer = MMUTrainer(model_name=model_name, subjects=subjects, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, gradient_accumulation_steps=gradient_accumulation_steps, output_dir=output_dir, accelerator=accelerator)

    # Log device information
    device = accelerator.device
    logger.info(f"Using device: {device}")

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on validation and test splits
    logger.info("Evaluating on validation split...")
    val_metrics = trainer.evaluate("validation")
    logger.info(f"Validation metrics: {val_metrics}")

    logger.info("Evaluating on test split...")
    test_metrics = trainer.evaluate("test")
    logger.info(f"Test metrics: {test_metrics}")

    logger.info("Training completed successfully!")
    except Exception as e: logger.error(f"Error during training: {str(e)}", exc_info=True)        raise


        if __name__ == "__main__":            main()
