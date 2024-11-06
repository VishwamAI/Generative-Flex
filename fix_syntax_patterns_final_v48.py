import os
import re

def fix_jax_trainer(content):
    """Fix syntax issues in jax_trainer.py."""
    # Fix module docstring
    content = re.sub(
        r'^""".*?"""',
        '"""JAX-based trainer implementation."""',
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    # Fix class definition and docstring
    content = re.sub(
        r'class JaxTrainer[^:]*:(\s*"""[^"]*""")?\s*',
        'class JaxTrainer:\n    """JAX trainer for model optimization."""\n\n',
        content
    )

    # Fix method definitions
    methods = {
        '__init__': 'Initialize the JAX trainer.',
        'train': 'Train the model using JAX optimization.',
        'evaluate': 'Evaluate the model performance.',
        'save_checkpoint': 'Save model checkpoint.',
        'load_checkpoint': 'Load model checkpoint.',
        'compute_loss': 'Compute training loss.',
        'forward_pass': 'Perform forward pass.',
        'backward_pass': 'Perform backward pass.',
        'optimize_step': 'Perform optimization step.',
    }

    for method, desc in methods.items():
        pattern = rf'def {method}\([^)]*\)(\s*->[\s\w\[\],]*)?:\s*(?:"""[^"]*""")?\s*'
        if method == '__init__':
            replacement = f'def {method}(self, model, optimizer, config):\n        """{desc}"""\n'
        else:
            replacement = f'def {method}(self, *args, **kwargs):\n        """{desc}"""\n'
        content = re.sub(pattern, replacement, content)

    return content

def fix_trainer(content):
    """Fix syntax issues in trainer.py."""
    # Fix module docstring
    content = re.sub(
        r'^""".*?"""',
        '"""Base trainer implementation."""',
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    # Fix class definition and docstring
    content = re.sub(
        r'class Trainer[^:]*:(\s*"""[^"]*""")?\s*',
        'class Trainer:\n    """Base trainer class for model training."""\n\n',
        content
    )

    # Fix method definitions with proper type hints
    methods = {
        '__init__': ('Initialize the trainer.', 'def __init__(self, model: torch.nn.Module, config: Any, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> None:'),
        'train': ('Train the model.', 'def train(self, epochs: int) -> None:'),
        'evaluate': ('Evaluate the model.', 'def evaluate(self) -> Dict[str, float]:'),
        'save_checkpoint': ('Save model checkpoint.', 'def save_checkpoint(self, path: str) -> None:'),
        'load_checkpoint': ('Load model checkpoint.', 'def load_checkpoint(self, path: str) -> None:'),
    }

    for method, (desc, signature) in methods.items():
        pattern = rf'def {method}\([^)]*\)(\s*->[\s\w\[\],]*)?:\s*(?:"""[^"]*""")?\s*'
        replacement = f'{signature}\n        """{desc}"""\n'
        content = re.sub(pattern, replacement, content)

    return content

def process_file(file_path):
    """Process a single file to fix syntax issues."""
    print(f"Processing {file_path}...")

    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'jax_trainer.py' in file_path:
        content = fix_jax_trainer(content)
    elif 'trainer.py' in file_path:
        content = fix_trainer(content)

    # Fix trailing whitespace and ensure single newline at end of file
    content = '\n'.join(line.rstrip() for line in content.splitlines())
    content = content.strip() + '\n'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Process trainer files to fix syntax issues."""
    files_to_fix = [
        "src/training/jax_trainer.py",
        "src/training/trainer.py"
    ]

    for file_path in files_to_fix:
        process_file(file_path)

if __name__ == "__main__":
    main()
