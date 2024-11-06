#!/usr/bin/env python3
"""Fix syntax issues by handling each file individually with specific patterns."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def fix_symbolic_math(content: str) -> str:
    """Fix syntax in symbolic_math.py."""
    # Fix class inheritance
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:',
        lambda m: f'class {m.group(1)}(nn.Module):',
        content
    )
    return content

def fix_text_to_anything(content: str) -> str:
    """Fix syntax in text_to_anything.py."""
    # Fix type hints
    content = re.sub(
        r'image_size:\s*Tuple\[int#\s*Training configuration',
        'image_size: Tuple[int, int]  # Training configuration',
        content
    )
    return content

def fix_train_mmmu(content: str) -> str:
    """Fix syntax in train_mmmu.py."""
    # Fix method signatures
    content = re.sub(
        r'r:\s*DataLoader\s*optimizer:\s*torch\.optim\.Optimizer,\s*config:\s*TrainingConfig\):',
        'dataloader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainingConfig):',
        content
    )
    return content

def fix_device_test(content: str) -> str:
    """Fix syntax in device_test.py."""
    # Fix multi-line statements
    content = re.sub(
        r'x\s*=\s*jnp\.ones\(\(1000,\s*1000\)\)',
        'x = jnp.ones((1000, 1000))',
        content
    )
    return content

def fix_test_environment(content: str) -> str:
    """Fix syntax in test_environment.py."""
    # Fix class inheritance
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*unittest\.TestCase\s*\)\s*:',
        lambda m: f'class {m.group(1)}(unittest.TestCase):',
        content
    )
    return content

def fix_training_logger(content: str) -> str:
    """Fix syntax in logging.py."""
    # Fix method definitions
    content = re.sub(
        r'class\s+TrainingLogger:\s*de,\s*f\s*log_dir:\s*str,\s*\(self,\s*log_dir:\s*str\s*=\s*"logs"\):\s*self,\s*\.log_dir\s*=\s*log_dir',
        'class TrainingLogger:\n    def __init__(self, log_dir: str = "logs"):\n        self.log_dir = log_dir',
        content
    )
    return content

def fix_timeout(content: str) -> str:
    """Fix syntax in timeout.py."""
    # Fix class inheritance
    content = re.sub(
        r'class\s+(\w+)\s*\(\s*Exception\s*\)\s*:\s*pas,\s*s',
        lambda m: f'class {m.group(1)}(Exception):\n    pass',
        content
    )
    return content

def fix_device_config(content: str) -> str:
    """Fix syntax in device_config.py."""
    # Fix method signatures
    content = re.sub(
        r'def\s+setup_device_config\(self\):\s*memory_fraction:\s*floa\s*=\s*0\.8\):\s*gpu_allow_growth:\s*boo,\s*l\s*=\s*True\s*\)\s*->\s*Dict\[str',
        'def setup_device_config(self, memory_fraction: float = 0.8, gpu_allow_growth: bool = True) -> Dict[str, Any]',
        content
    )
    return content

def fix_simple_model(content: str) -> str:
    """Fix syntax in simple_model.py."""
    # Fix parameter definitions
    content = re.sub(
        r'vocab_size:\s*inthidden_dim:\s*int\s*=\s*32',
        'vocab_size: int, hidden_dim: int = 32',
        content
    )
    return content

def fix_video_model(content: str) -> str:
    """Fix syntax in video_model.py."""
    # Fix type hints
    content = re.sub(
        r'int\]#\s*\(time\s*heightwidth\)',
        'int]  # (time, height, width)',
        content
    )
    return content

def fix_train_chatbot(content: str) -> str:
    """Fix syntax in train_chatbot.py."""
    # Fix method signatures
    content = re.sub(
        r'def\s+load_data\(self\):\s*file_path:\s*st\s*=\s*"data/chatbot/training_data_cot\.json"\)\s*->\s*List\[Dict\[str\):\s*str,\s*\]\]:',
        'def load_data(self, file_path: str = "data/chatbot/training_data_cot.json") -> List[Dict[str, str]]:',
        content
    )
    return content

def process_file(file_path: Path) -> None:
    """Process a single file with specific fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply specific fixes based on filename
        if file_path.name == 'symbolic_math.py':
            content = fix_symbolic_math(content)
        elif file_path.name == 'text_to_anything.py':
            content = fix_text_to_anything(content)
        elif file_path.name == 'train_mmmu.py':
            content = fix_train_mmmu(content)
        elif file_path.name == 'device_test.py':
            content = fix_device_test(content)
        elif file_path.name == 'test_environment.py':
            content = fix_test_environment(content)
        elif file_path.name == 'logging.py':
            content = fix_training_logger(content)
        elif file_path.name == 'timeout.py':
            content = fix_timeout(content)
        elif file_path.name == 'device_config.py':
            content = fix_device_config(content)
        elif file_path.name == 'simple_model.py':
            content = fix_simple_model(content)
        elif file_path.name == 'video_model.py':
            content = fix_video_model(content)
        elif file_path.name == 'train_chatbot.py':
            content = fix_train_chatbot(content)

        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main() -> None:
    """Process all Python files in the project."""
    # Get all Python files
    python_files = []
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        python_files.extend(Path(".").glob(pattern))

    # Process each file
    for file_path in python_files:
        if not any(part.startswith('.') for part in file_path.parts):
            process_file(file_path)

if __name__ == "__main__":
    main()
