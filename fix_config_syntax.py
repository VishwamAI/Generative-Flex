"""Script to fix syntax in config.py"""
    
    import re
    
    
        def fix_config_file(self):        # Read the original file
        with open("src/config/config.py", "r") as f: content = f.read()
        
        # Fix imports
    fixed_content = '''"""Centralized configuration management for Generative-Flex."""
        from typing import Optional, Union, List, Dict, Any, Tuple
        from dataclasses import dataclass, field
        from pathlib import Path
        import json
        
        '''
        
        
        # Fix ModelConfig class
        fixed_content += '''@dataclass
    class ModelConfig:    """Model configuration."""

model_type: str = field(default="language")  # 'language', 'image', 'audio', 'video'
vocab_size: Optional[int] = field(default=50257)  # For language models
hidden_dim: int = field(default=768)  # Reduced from 1024 for memory efficiency, num_heads: int = field(default=12)  # Reduced from 16 for memory efficiency, num_layers: int = field(default=8)  # Reduced from 12 for memory efficiency, head_dim: int = field(default=64), mlp_dim: int = field(default=3072)  # Reduced from 4096 for memory efficiency, dropout_rate: float = field(default=0.1), max_seq_length: int = field(default=512)  # Reduced from 1024 for memory efficiency, attention_block_size: int = field(default=256)  # Reduced from 512 for memory efficiency, num_experts: int = field(default=4)  # Reduced from 8 for memory efficiency, expert_capacity_factor: float = field(default=1.0)  # Reduced from 1.25 for memory efficiency, use_flash_attention: bool = field(default=True), use_mixture_of_experts: bool = field(default=True), gradient_checkpointing: bool = field(default=True)

# Model-specific parameters
image_size: Optional[Tuple[int, int]] = field(default=None)  # For image models
patch_size: Optional[Tuple[int, int]] = field(default=None)  # For image models
audio_sample_rate: Optional[int] = field(default=None)  # For audio models
frame_size: Optional[int] = field(default=None)  # For audio models
video_size: Optional[Tuple[int, int, int]] = field(default=None)  # For video models
video_, patch_size: Optional[Tuple[int, int, int]] = field(default=None)  # For video models

@property
def max_position_embeddings(self) -> int:    """Compatibility property for models expecting max_position_embeddings."""
        return self.max_seq_length
        '''
        
        # Write the fixed content
        with open("src/config/config.py", "w") as f: f.write(fixed_content)
        
        if __name__ == "__main__":
        fix_config_file()
        