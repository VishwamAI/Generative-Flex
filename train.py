from src.training.train_mmmu import MMUTrainer
import argparse
import logging
import os



def parse_args(self):
    parser = argparse.ArgumentParser(description="Train MMMU model with enhanced mathematical reasoning")
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b", help="Base model to use (default: facebook/opt-1.3b)")
    parser.add_argument("--subjects", type=str, default="Math", help="Subjects to train on (default: Math)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Number of gradient accumulation steps (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length (default: 512)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps (default: 500)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps (default: 100)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    return parser.parse_args()


def main(self):
    args = parse_args()

    # Set up logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
    logging.StreamHandler(),
    logging.FileHandler("logs/training.log"),
    ])
    logger = logging.getLogger(__name__)

    # Log configuration
    logger.info("Training configuration:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

        # Initialize trainer
        trainer = MMUTrainer(model_name=args.model_name, subjects=[args.subjects], batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, num_epochs=args.num_epochs, output_dir=args.output_dir)

        try:
            # Start training
            logger.info("Starting training...")
            trainer.train()

            except Exception as e: logger.error(f"Training failed with error: {str(e)}")
                raise

                logger.info("Training completed successfully!")


                if __name__ == "__main__":
                    main()
