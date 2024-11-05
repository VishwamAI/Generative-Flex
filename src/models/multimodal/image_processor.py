"""
Image processor for handling multimodal inputs in the MMMU model.
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ImageProcessor(nn.Module):
    """Process multiple images for multimodal transformer input."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # CNN backbone for processing individual images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # Output shape: [batch_size * num_images, 64]
        )

        # Project CNN features to transformer hidden size
        self.projector = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process batch of images.
        Args:
            images: Tensor of shape [batch_size, channels, height, width] or
                   [batch_size, num_images, channels, height, width]
        Returns:
            Tensor of shape [batch_size, num_images, hidden_size] or
            [batch_size, 1, hidden_size] for single images
        """
        try:
            # Handle both 4D and 5D inputs
            if len(images.shape) == 4:
                batch_size, channels, height, width = images.shape
                num_images = 1
                # Add num_images dimension
                images = images.unsqueeze(1)
            elif len(images.shape) == 5:
                batch_size, num_images, channels, height, width = images.shape
            else:
                raise ValueError(f"Expected 4D or 5D input tensor, got shape {images.shape}")

            # Log input shape for debugging
            logger.info(f"Processing image chunk {batch_size}/{num_images}, shape: {images.shape}")

            # Reshape for CNN processing
            flat_images = images.view(-1, channels, height, width)

            # Process through CNN
            cnn_features = self.cnn(flat_images)  # Shape: [batch_size * num_images, 64]

            # Project to hidden size
            projected = self.projector(cnn_features)  # Shape: [batch_size * num_images, hidden_size]

            # Reshape back to [batch_size, num_images, hidden_size]
            output = projected.view(batch_size, num_images, self.hidden_size)

            # Log output shape for debugging
            logger.info(f"Final processed image shape: {output.shape}")

            return output

        except Exception as e:
            logger.error(f"Error in ImageProcessor forward pass: {str(e)}")
            # Return zero tensor of correct shape with proper device
            if len(images.shape) == 4:
                return torch.zeros(images.size(0), 1, self.hidden_size, device=images.device)
            else:
                return torch.zeros(images.size(0), images.size(1), self.hidden_size, device=images.device)
