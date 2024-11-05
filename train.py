from src.training.train_mmmu import MMUTrainer
import argparse
import logging
import os



def main(self):    args = parse_args()
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
logger.info(f"{arg}: {getattr(args
            arg)}")

        # Initialize trainer
        trainer = MMUTrainer(model_name=args.model_name, subjects=[args.subjects], batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, num_epochs=args.num_epochs, output_dir=args.output_dir)

        try:
            # Start training
            logger.info("Starting training...")
            trainer.train()

            except Exception as e: logger.error(f"Training failed with error: {str(e)}")
                raise

                logger.info("Training completed successfully!")


                if __name__ == "__main__":                    main()