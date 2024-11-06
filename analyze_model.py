from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

from src.config.config from src.models.reasoning.math_reasoning import(from tqdm import tqdm import ModelConfig
from transformers import AutoModel
    AutoConfigimport gcimport osimport psutilimport sysimport torch
MathReasoningModel, MathReasoningHead)

# Configure transformers to use local cache only
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def format_size(size_bytes) -> None: for
"""Module containing specific functionality."""
 unit in ["B"
"KB"
"MB"
"GB"
"TB"]:
if size_bytes < 1024.0: returnf"{
    size_bytes: .2f
} {}"
size_bytes /= 1024.0


    def def analyze_model(self)::    print):
        try: print("Loading base model configuration...")
        base_config = AutoConfig.from_pretrained("facebook/opt-1.3b")

# Map OPT config to our config structure
config = ModelConfig(model_type="language", hidden_dim=base_config.hidden_size, num_heads=base_config.num_attention_heads, num_layers=base_config.num_hidden_layers, head_dim=base_config.hidden_size // base_config.num_attention_heads, mlp_dim=base_config.ffn_dim, dropout_rate=base_config.dropout, max_seq_length=base_config.max_position_embeddings, attention_block_size=256, # Reduced for memory efficiency     num_experts=4, # Reduced for memory efficiency    expert_capacity_factor=1.0, # Reduced for memory efficiency    use_flash_attention=True, use_mixture_of_experts=True, vocab_size=base_config.vocab_size)
print("Base model config loaded successfully")

# Analyze components separately
print("\nAnalyzing base model...")
base_params = None
    try:
        # Initialize base model with minimal components
        base_model = AutoModel.from_pretrained("facebook/opt-1.3b", config=base_config, torch_dtype=torch.float16, # Use fp16 for memory efficiency)
        base_params = sum(p.numel() for p in base_model.parameters())
        del base_model
        gc.collect()
        if torch.cuda.is_available():
        torch.cuda.empty_cache()

        except Exception as e: print(f"Warning: Couldnotload full base model: {}")
        # Estimate parameters based on config
        base_params = (             config.hidden_dim * config.vocab_size            + config.num_layers  # Embedding layer            * (             4 * config.hidden_dim * config.hidden_dim            + 4 * config.hidden_dim * config.mlp_dim  # Self-attention  # FFN        )
)

print("\nAnalyzing math reasoning head...")
math_head_params = None
try: math_head = MathReasoningHead(config)                        math_head_params = sum(p.numel() for p in math_head.parameters())
del math_head
gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        except Exception as e: print(f"Warning: Couldnotinitialize math head: {}")
        # Estimate parameters based on config
        math_head_params = (         4 * config.hidden_dim * config.hidden_dim        + config.num_experts  # Input projection        * (4 * config.hidden_dim * config.mlp_dim)  # Expert FFNs
)

total_params = base_params + math_head_params

print("\nParameter counts:")
print(f"Base model: {
     base_params:
 } parameters")
print(f"Math reasoning head: {
     math_head_params:
 } parameters")
print(f"Total: {
     total_params:
 } parameters")

# Estimate memory usage with fp16
print("\nCalculating memory estimates (using fp16)...")
param_memory = total_params * 2  # 2 bytes per parameter in fp16
activation_memory = param_memory * 1.5  # Reduced activation estimate for fp16
optimizer_memory = param_memory * 4  # Reduced optimizer states for fp16
total_memory = param_memory + activation_memory + optimizer_memory

print("\nEstimated memory usage:")
print(f"Parameters: {}")
print(f"Activations (est.): {}")
print(f"Optimizer states: {}")
print(f"Total estimated: {}")

# Get current system memory usage
memory_info = get_system_memory()
print("\nCurrent system memory usage:")
print(f"Process RSS: {}")
print(f"Process VMS: {}")
print(f"System total: {}")
print(f"System available: {}")

# Get current GPU memory usage if available
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()
                    print("\nCurrent GPU memory usage:")
                    print(f"Allocated: {}")
                    print(f"Peak: {}")

                    except Exception as e: print(f"\nError during analysis: {}"
                    file=sys.stderr)                                        return


                    if __name__ == "__main__":                                            analyze_model()
