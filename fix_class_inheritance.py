#!/usr/bin/env python3
"""Fix class inheritance and method signature syntax issues."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def fix_nn_module_class(content: str) -> str:
    """Fix nn.Module class definitions and their __init__ methods."""
    patterns = [
        # Fix class with vocab_size and hidden_size
        (r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:\s*vocab_size:\s*int,\s*hidden_size:\s*int\s*=\s*64',
         lambda m: f'''class {m.group(1)}(nn.Module):
    """Neural network module for {m.group(1)}."""

    def __init__(self, vocab_size: int, hidden_size: int = 64):
        """Initialize the module.

        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size'''),

        # Fix class with only hidden_size
        (r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:\s*hidden_size:\s*int\s*=\s*64',
         lambda m: f'''class {m.group(1)}(nn.Module):
    """Neural network module for {m.group(1)}."""

    def __init__(self, hidden_size: int = 64):
        """Initialize the module.

        Args:
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.hidden_size = hidden_size'''),

        # Fix basic nn.Module class
        (r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)\s*:(\s*$|\s+[^\n])',
         lambda m: f'''class {m.group(1)}(nn.Module):
    """Neural network module for {m.group(1)}."""

    def __init__(self):
        """Initialize the module."""
        super().__init__(){m.group(2)}''')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    return content

def fix_unittest_class(content: str) -> str:
    """Fix unittest.TestCase class definitions."""
    pattern = r'class\s+(\w+)\s*\(\s*unittest\.TestCase\s*\)\s*:'
    replacement = lambda m: f'''class {m.group(1)}(unittest.TestCase):
    """Test cases for {m.group(1)}."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()'''
    return re.sub(pattern, replacement, content)

def fix_train_state_class(content: str) -> str:
    """Fix train_state.TrainState class definitions."""
    pattern = r'class\s+(\w+)\s*\(\s*train_state\.TrainState\s*\)\s*:'
    replacement = lambda m: f'''class {m.group(1)}(train_state.TrainState):
    """Custom training state for {m.group(1)}."""

    def __init__(self, *args, **kwargs):
        """Initialize the training state."""
        super().__init__(*args, **kwargs)'''
    return re.sub(pattern, replacement, content)

def fix_method_signatures(content: str) -> str:
    """Fix method signatures and their docstrings."""
    patterns = [
        # Fix forward method
        (r'def\s+forward\s*\(\s*self,\s*([^)]*)\)\s*:\s*\*\*kwargs\):\s*Forwar,\s*d\s*pass',
         lambda m: f'''def forward(self, {m.group(1)}, **kwargs):
        """Forward pass through the network.

        Args:
            {", ".join(arg.strip().split(":")[0] + ": " + arg.strip().split(":")[-1].strip() for arg in m.group(1).split(",") if arg.strip())}
            **kwargs: Additional arguments

        Returns:
            Network output
        """'''),

        # Fix setup_device_config method
        (r'def\s+setup_device_config\s*\(\s*self,\s*memory_fraction:\s*float\s*=\s*0\.8,\s*gpu_allow_growth:\s*bool\s*=\s*True\s*\)\s*->\s*Dict\[str,\s*Any\]',
         lambda m: '''def setup_device_config(
        self,
        memory_fraction: float = 0.8,
        gpu_allow_growth: bool = True
    ) -> Dict[str, Any]:
        """Set up device configuration.

        Args:
            memory_fraction: Fraction of GPU memory to allocate
            gpu_allow_growth: Whether to allow GPU memory growth

        Returns:
            Dict containing device configuration
        """'''),

        # Fix load_data method
        (r'def\s+load_data\s*\(\s*self,\s*file_path:\s*str\s*=\s*"[^"]+"\s*\)\s*->\s*List\[Dict\[str,\s*str\]\]:\s*wit,\s*h',
         lambda m: '''def load_data(
        self,
        file_path: str = "data/chatbot/training_data_cot.json"
    ) -> List[Dict[str, str]]:
        """Load training data from file.

        Args:
            file_path: Path to training data file

        Returns:
            List of conversation dictionaries
        """
        with''')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def fix_type_hints(content: str) -> str:
    """Fix type hint formatting."""
    patterns = [
        # Fix Tuple type hints
        (r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*):\s*Tuple\[([^\]]+)\](\s*#[^\n]*)?',
         lambda m: f'{m.group(1)}{m.group(2)}: Tuple[{m.group(3).replace(" ", "")}]{m.group(4) if m.group(4) else ""}'),

        # Fix Dict type hints
        (r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*):\s*Dict\[([^\]]+)\](\s*=\s*[^,\n]+)?',
         lambda m: f'{m.group(1)}{m.group(2)}: Dict[{m.group(3).replace(" ", "")}]{m.group(4) if m.group(4) else ""}'),

        # Fix List type hints
        (r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*):\s*List\[([^\]]+)\](\s*=\s*[^,\n]+)?',
         lambda m: f'{m.group(1)}{m.group(2)}: List[{m.group(3).replace(" ", "")}]{m.group(4) if m.group(4) else ""}')
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    return content

def process_file(file_path: Path) -> None:
    """Process a single file with all fixes."""
    print(f"Processing {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply all fixes
        content = fix_nn_module_class(content)
        content = fix_unittest_class(content)
        content = fix_train_state_class(content)
        content = fix_method_signatures(content)
        content = fix_type_hints(content)

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
