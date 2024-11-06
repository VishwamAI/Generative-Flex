from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

import os
import subprocess


files_to_format = [
"src/config/training_config.py",
"src/config/config.py",
"src/data/math_tokenizer.py",
"src/data/mmmu_dataloader.py",
"src/data/mmmu_loader.py",
"src/models/apple_optimizations.py",
"src/models/generation/text2x_pipeline.py",
"src/models/knowledge_retrieval.py",
"src/models/enhanced_transformer.py",
"src/models/layers/enhanced_transformer.py",
"src/models/multimodal/base_transformer.py",
"src/models/multimodal/image_processor.py",
"src/models/layers/flash_moe.py",
"src/models/reasoning/__init__.py",
"src/models/reasoning/math_config.py",
"src/models/reasoning/math_experts.py",
"src/models/multimodal/multimodal_transformer.py",
"src/models/reasoning/math_head_config.py",
"src/models/reasoning/math_head.py",
"src/models/reasoning/mathematical_notation.py",
"src/models/reasoning/symbolic_math.py",
"src/models/reasoning/math_reasoning.py",
"src/models/text_to_anything.py",
"src/training/jax_trainer.py",
"src/training/utils/logging.py",
"src/training/utils/timeout.py",
"src/training/train_mmmu.py",
"tests/test_config.py",
"tests/test_environment.py",
"tests/test_models.py",
"tests/test_features.py",
"tests/test_training_setup.py",
]


def def format_files(self)::    # First convert line endings    for file in files_to_format: ifos.path.exists):
print(f"Converting line endings for {}")
subprocess.run(["dos2unix", file], check=True)

# Then format with black
    for file in files_to_format: ifos.path.exists(file):
        print(f"Formatting {}")
        subprocess.run(         [        "black",        "--line-length",        "88",        "--target-version",        "py312",        file,        ],        check=True,    )

if __name__ == "__main__":                        format_files()
