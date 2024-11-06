import re
import os

def fix_enhanced_transformer():
    print("Fixing enhanced_transformer.py...")
    file_path = "src/models/layers/enhanced_transformer.py"

    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix docstring syntax
    content = re.sub(
        r'"""Enhanced transformer layer with advanced features\."""',
        '"""Enhanced transformer layer implementation with advanced features."""',
        content
    )

    # Fix class definition and imports
    content = re.sub(
        r'class EnhancedTransformer\(.*?\):',
        'class EnhancedTransformer(nn.Module):',
        content,
        flags=re.DOTALL
    )

    # Ensure proper import statements
    imports = """from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
"""

    if not any(imp in content for imp in ['import torch', 'from torch']):
        content = imports + "\n" + content

    with open(file_path, 'w') as f:
        f.write(content)

def fix_test_models():
    print("Fixing test_models.py...")
    file_path = "src/tests/test_models.py"

    if not os.path.exists(file_path):
        print(f"Checking alternative path...")
        file_path = "tests/test_models.py"
        if not os.path.exists(file_path):
            print(f"File test_models.py not found!")
            return

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix imports and add proper test class structure
    fixed_content = """import unittest
import torch
import torch.nn as nn
from src.models.transformer import TransformerModel
from src.config.config import ModelConfig

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128
        )

    def test_transformer_model(self):
        model = TransformerModel(self.config)
        self.assertIsInstance(model, nn.Module)

    def test_model_forward(self):
        model = TransformerModel(self.config)
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        outputs = model(input_ids)
        self.assertEqual(outputs.shape, (batch_size, seq_length, self.config.hidden_size))

if __name__ == '__main__':
    unittest.main()
"""

    with open(file_path, 'w') as f:
        f.write(fixed_content)

if __name__ == "__main__":
    fix_enhanced_transformer()
    fix_test_models()
