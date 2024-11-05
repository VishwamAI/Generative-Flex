"""
Training script for MMMU dataset with enhanced model architecture.
"""

import gc
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, AutoConfig
from accelerate import Accelerator

from src.models.enhanced_transformer import EnhancedTransformer
from src.models.multimodal.multimodal_transformer import MultiModalTransformer
from src.models.reasoning.math_head import MathReasoningHead
from src.models.multimodal.image_processor import ImageProcessor

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Default subjects for MMMU dataset
DEFAULT_SUBJECTS = ["Math", "Computer_Science"]


def log_metrics(
metrics: Dict[str, float], step: Optional[int] = None, epoch: Optional[int] =
    None
):
    """Log training metrics to console and file"""
    metric_str = " - ".join(
        [
            f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}"
            for k, v in metrics.items()
        ]
    )
    if epoch is not None:
        metric_str = f"Epoch {epoch} - {metric_str}"
    if step is not None:
        metric_str = f"Step {step} - {metric_str}"
    logger.info(metric_str)


# Define MMMU subjects and splits
MMMU_SUBJECTS = ["Math", "Computer_Science"] # Exact subject names from MMMU
dataset
MMMU_SPLITS = ["dev", "validation", "test"]  # Available splits in MMMU


class EnhancedMMUModel(PreTrainedModel):
supports_gradient_checkpointing = True # Enable gradient checkpointing support

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 4  # A, B, C, D options
        self.config = config

# Set model dimensions for mathematical reasoning while maintaining efficiency
        config.hidden_size = min(
            config.hidden_size, 256
        )  # Reduced for memory efficiency
        config.num_attention_heads = min(
            config.num_attention_heads, 4
        )  # Fewer heads for efficiency
        config.num_hidden_layers = min(
            config.num_hidden_layers,
            3)  # Reduced layers)

        # Base transformer with enhanced attention
        self.transformer = EnhancedTransformer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            vocab_size=config.vocab_size,
        )

        # Multimodal processing with reduced size
        self.multimodal = MultiModalTransformer(config)

        # Math reasoning specific components
        self.math_head = MathReasoningHead(config)

        # Image processing components
        self.image_processor = ImageProcessor(hidden_size=config.hidden_size)

        # Fusion layer for combining modalities
        self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # Initialize weights
        self.init_weights()

    def process_images(self, images: torch.Tensor) -> torch.Tensor:
        """Process batch of images with proper error handling and reshaping"""
        try:
            if images is None:
                return None

            if not isinstance(images, torch.Tensor):
                return None

            batch_size = images.size(0)
            logger.info(
                f"Processing image chunk 0/{batch_size},
                shape: {images.shape}")

            # Ensure images are in the correct format
            if (
                len(images.shape) != 5
            ):  # [batch_size, num_images, channels, height, width]
                raise ValueError(
                    f"Expected 5D input tensor,
                    got shape {images.shape}")

            # Process images through the image processor
            image_features = self.image_processor(images)
            return image_features

        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")
            # Return zero tensor of correct shape
            return torch.zeros(images.size(0), 7, self.config.hidden_size)

    def training_step(self, batch: dict) -> torch.Tensor:
        """Execute single training step with proper tensor handling"""
        try:
            # Process text inputs
            transformer_outputs = self.transformer(
input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            hidden_states = transformer_outputs["last_hidden_state"]

            # Process images if present
            if "images" in batch:
                image_features = self.process_images(batch["images"])
                # Combine text and image features
                combined_features = torch.cat(
                    [hidden_states,
                    image_features],
                    dim=-1)
                fused_features = self.fusion(combined_features)
            else:
                fused_features = hidden_states

            # Get math reasoning outputs
            math_outputs = self.math_head(
                fused_features,
                batch["attention_mask"])

            # Calculate losses
            if "labels" in batch:
                loss_fct = nn.CrossEntropyLoss()
                task_loss = loss_fct(
                    math_outputs["logits"].view(-1, self.num_labels),
                    (batch["labels"].view(-1),),
                )
                moe_loss = math_outputs.get(
                    "moe_loss", torch.tensor(0.0, device=task_loss.device)
                )
                total_loss = task_loss + 0.01 * moe_loss
                return total_loss

            return torch.tensor(0.0, requires_grad=True)

        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            # Return a default loss that can be backpropagated
            return torch.tensor(float("inf"), requires_grad=True)

        # Enable gradient checkpointing if specified in config
        if getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency"""
        self.transformer.gradient_checkpointing = True

    def _set_gradient_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing."""
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable(enable)
        if hasattr(self.multimodal, "gradient_checkpointing_enable"):
            self.multimodal.gradient_checkpointing_enable(enable)
        if hasattr(self.math_head, "gradient_checkpointing_enable"):
            self.math_head.gradient_checkpointing_enable(enable)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass with proper error handling and memory management."""
        try:
            # Get batch size from input_ids
            batch_size = input_ids.size(0) if input_ids is not None else 0
            chunk_size = 1  # Process one image at a time if memory is limited

            # Process images if provided
            if images is not None:
                logger.info(f"Starting image processing with batch size {
                    batch_size}")}
                try:
                    # Try processing all images at once
                    processed_images = self.process_images(images)
                except RuntimeError as e:
                    logger.warning(f"Memory error in batch processing: {
                        str(e)}")}
                    # Fall back to chunk processing
                    processed_chunks = []
                    for i in range(0, batch_size, chunk_size):
                        chunk = images[i: i + chunk_size]
                        logger.info(
f"Processing image chunk {i}/{batch_size}, shape: {chunk.shape}"
                        )
                        try:
                            processed_chunk = self.process_images(chunk)
                            processed_chunks.append(processed_chunk)
                            torch.cuda.empty_cache()
                        except RuntimeError as e:
                            logger.error(f"Error processing chunk {
                                i}: {str(e)}")}
                            return None
                        else:
                            gc.collect()
                    processed_images = torch.cat(processed_chunks, dim=0)
            else:
                processed_images = None

            # Get embeddings from transformer
            hidden_states = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=processed_images,
            )

            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()

            # Get math outputs
            math_outputs = self.math_head(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Combine outputs
            result = {
                "logits": math_outputs["logits"],
                "router_entropy": math_outputs.get("router_entropy", 0.0),
                "expert_weights": math_outputs.get("expert_weights", None),
                "last_hidden_state": hidden_states,
                "loss": None,  # Initialize loss as None
            }

            if labels is not None:
                result["loss"] = math_outputs["loss"]

            return result

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise


class MMUTrainer:
    """Trainer class for MMMU model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1,
        learning_rate: float = 5e-6,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 32,
        max_grad_norm: float = 0.1,
        warmup_steps: int = 100,
        generation_config: Optional[Dict] = None,
        output_dir: str = "outputs",
        config=None,
        tokenizer=None,
    ):
        """Initialize the MMU trainer with Accelerate support."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
# self.batch_size = batch_size # TODO: Remove or use this variable
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.generation_config = generation_config or {}
        self.output_dir = output_dir
        self.tokenizer = tokenizer

        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            device_placement=True,
            mixed_precision="fp16" if torch.cuda.is_available() else None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_dir=output_dir,
        )

        # Set up model components with proper config
        if config is None:
            config = AutoConfig.from_pretrained("bert-base-uncased")
# config.hidden_size = 256 # TODO: Remove or use this variable
            config.num_attention_heads = 4
            config.num_hidden_layers = 3
            config.intermediate_size = 512
            config.max_position_embeddings = 512
            config.gradient_checkpointing = True
            config.use_cache = False

        # Initialize model with config
        if not isinstance(model, nn.Module):
            self.model = EnhancedMMUModel(config)
            self.model.gradient_checkpointing_enable()

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(
                    ),
                    lr=learning_rate,
                    weight_decay=0.05,
                    eps=1e-8)
            )

        # Create learning rate scheduler if not provided
        if scheduler is None:
            num_training_steps = (
                1000  # Default value, should be updated with dataloader length
            )
            num_warmup_steps = min(
                warmup_steps,
                int(num_training_steps * 0.15)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        # Move model to device
        self.model = self.model.to(self.device)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized MMUTrainer with {self.device} device")
        logger.info(
            f"Model parameters: {sum(
                p.numel() for p in self.model.parameters()):,
                }")
        )

    def train(self):
        """Training loop with proper error handling and logging"""
logger.info("Starting training with mixed precision and gradient accumulation")
        self.model.train()

        # Initialize progress tracking
        total_steps = len(self.train_dataloader) * self.num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")

        best_val_loss = float("inf")
        best_math_acc = 0.0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.train_dataloader):
                try:
                    # Forward pass with gradient accumulation
                    with self.accelerator.accumulate(self.model):
                        # Process batch and compute loss
                        outputs = self.model(**batch)

                        # Handle loss calculation
                        if isinstance(outputs, dict):
                            loss = outputs.get("loss")
                            if (
                                loss is None
                                and "logits" in outputs
                                and "labels" in batch
                            ):
                                # Calculate loss if not provided by model
                                loss_fct = torch.nn.CrossEntropyLoss()
                                logits = outputs["logits"]
                                loss = loss_fct(
                                    logits.view(-1, self.model.num_labels),
                                    batch["labels"].view(-1),
                                )
                        else:
                            loss = outputs.loss if hasattr(
                                outputs,
                                "loss") else None)

                        if loss is None:
                            continue

                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps

                        # Backward pass with mixed precision
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Update progress
epoch_loss += loss.item() * self.gradient_accumulation_steps
                        num_batches += 1

                        # Log training metrics
                        if step % self.gradient_accumulation_steps == 0:
                            current_lr = self.optimizer.param_groups[0]["lr"]
                            metrics = {
                                "loss": loss.item(
                                    ) * self.gradient_accumulation_steps,
                                    )
                                "learning_rate": current_lr,
                            }
                            if (
                                isinstance(outputs, dict)
                                and "logits" in outputs
                                and "labels" in batch
                            ):
                                predictions = torch.argmax(
                                    outputs["logits"],
                                    dim=-1)
                                correct = (predictions == batch[
                                    "labels"]).sum().item()]
                                total = batch["labels"].size(0)
                                metrics["batch_accuracy"] = correct / total

                            log_metrics(metrics, step=step, epoch=epoch)

                        progress_bar.update(1)

                except Exception as e:
                    logger.error(f"Error in training step: {str(e)}")
                    continue

            # Skip epoch summary if no batches were processed
            if num_batches == 0:
                logger.warning(
                    f"Epoch {epoch} had no valid batches, skipping evaluation"
                )
                continue

            # Compute average loss for epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

            # Evaluate on validation set
            val_metrics = self.evaluate()
            val_loss = val_metrics["val_loss"]
            math_acc = val_metrics["math_accuracy"]

            # Log validation metrics
            logger.info(f"Validation loss: {val_loss:.4f}")
            logger.info(f"Validation math accuracy: {math_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.accelerator.save_state("outputs/best_loss_checkpoint")

            if math_acc > best_math_acc:
                best_math_acc = math_acc
                self.accelerator.save_state("outputs/best_math_checkpoint")

            # Save training state for recovery
            self.accelerator.save_state("outputs/latest_checkpoint")
        progress_bar.close()
        logger.info("Training completed")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best math accuracy: {best_math_acc:.4f}")

    def evaluate(self, eval_dataloader):
        """Evaluation loop with proper error handling and logging."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=batch.get("images", None),
                    labels=batch.get("labels", None),
                )

                # Calculate metrics
                loss = outputs["loss"] if "loss" in outputs else 0
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == batch["labels"]).sum().item()

                # Update totals
                total_loss += loss.item()
                total_correct += correct
                total_samples += batch["labels"].numel()

        # Calculate final metrics
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = total_correct / total_samples

        return {"eval_loss": avg_loss, "eval_accuracy": accuracy}


if __name__ == "__main__":
    try:
        # Initialize trainer with optimized settings for local hardware
        trainer = MMUTrainer(
model_name="facebook/opt-1.3b", # Use larger OPT model for better capacity
            subjects=["Math"],  # Focus solely on mathematical reasoning
            batch_size=1,  # Smaller batch size for larger model
            learning_rate=1e-5,  # Lower learning rate for stability
            num_epochs=5,
gradient_accumulation_steps=8, # More gradient accumulation for larger
            effective batch
            output_dir="outputs",
        )

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        trainer.model.to(device)

        # Train the model
        print("Starting training...")
        trainer.train()

        # Evaluate on validation and test splits
        print("Evaluating on validation split...")
        val_metrics = trainer.evaluate("validation")
        print("Validation metrics:", val_metrics)

        print("Evaluating on test split...")
        test_metrics = trainer.evaluate("test")
        print("Test metrics:", test_metrics)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
