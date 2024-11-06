import os
import re

def fix_imports_and_docstrings(content):
    """Fix import statements and docstrings."""
    # Fix transformers import with GenerationMixin
    content = re.sub(
        r'from transformers import PreTrainedModel import GenerationMixin',
        'from transformers import PreTrainedModel, GenerationMixin',
        content
    )

    # Fix dataclass imports with docstring
    content = re.sub(
        r'from """[^"]+""" import dataclasses import dataclass, field',
        '"""Configuration for text-to-anything generation."""\nfrom dataclasses import dataclass, field',
        content
    )

    # Fix enhanced_transformer import with logger
    content = re.sub(
        r'from src\.models\.enhanced_transformer import EnhancedTransformer import logger',
        'from src.models.enhanced_transformer import EnhancedTransformer\nfrom src.utils.logging import logger',
        content
    )

    # Fix json import with nn.Module
    content = re.sub(
        r'import json\(nn\.Module\):',
        'import json\n\nclass SimpleModel(nn.Module):\n    """Simple model class."""',
        content
    )

    return content

def fix_class_definitions(content):
    """Fix class definitions and inheritance."""
    # Fix nn.Module inheritance with docstring
    content = re.sub(
        r'\(nn\.Module\):\s*$',
        '(nn.Module):\n    """Base model class."""\n\n    def __init__(self):\n        super().__init__()',
        content,
        flags=re.MULTILINE
    )

    # Fix unittest.TestCase inheritance
    content = re.sub(
        r'\(unittest\.TestCase\):\s*$',
        '(unittest.TestCase):\n    """Test case class."""\n\n    def setUp(self):\n        super().setUp()',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_method_definitions(content):
    """Fix method definitions and parameters."""
    # Fix parameter definitions
    content = re.sub(
        r'(\w+):\s*(\w+)\s*=\s*(\d+)',
        r'\1: \2 = \3',
        content
    )

    # Fix docstring placement
    content = re.sub(
        r'^\s*"""([^"]+)"""',
        lambda m: f'    """{m.group(1).strip()}."""',
        content,
        flags=re.MULTILINE
    )

    # Fix method definitions with type hints
    content = re.sub(
        r'def\s+(\w+)\s*\(\s*([^)]*)\)\s*->\s*None:\s*([^:]+)',
        lambda m: f'def {m.group(1)}({m.group(2).strip()}) -> None:\n    """{m.group(3).strip()}."""',
        content,
        flags=re.MULTILINE
    )

    return content

def fix_indentation_and_spacing(content):
    """Fix indentation and spacing issues."""
    # Fix indentation of class methods
    content = re.sub(
        r'^(\s*)def\s+',
        r'    \1def ',
        content,
        flags=re.MULTILINE
    )

    # Fix spacing around type hints
    content = re.sub(
        r'(\w+)\s*:\s*(\w+)',
        r'\1: \2',
        content
    )

    # Fix multiline string indentation
    content = re.sub(
        r'^\s*"""\s*-',
        r'    """-',
        content,
        flags=re.MULTILINE
    )

    return content

def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes in sequence
        content = fix_imports_and_docstrings(content)
        content = fix_class_definitions(content)
        content = fix_method_definitions(content)
        content = fix_indentation_and_spacing(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {filepath}")
        else:
            print(f"No changes needed for {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

def main():
    """Process files with syntax errors."""
    problem_files = [
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/simple_model.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/models/video_model.py',
        'src/test_inference.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_cot_fixed.py',
        'src/train_chatbot.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/jax_trainer.py',
        'src/training/accelerated_trainer.py',
        'src/training/trainer.py',
        'src/training/train_mmmu.py',
        'src/training/utils/timeout.py',
        'src/training/utils/logging.py'
    ]

    print(f"Processing {len(problem_files)} files...")
    for filepath in problem_files:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"Warning: {filepath} does not exist")

if __name__ == '__main__':
    main()
