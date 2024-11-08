import re

def fix_math_head_config():
    # Read the current content
    with open('src/models/reasoning/math_head_config.py', 'r') as f:
        content = f.read()

    # Fix imports
    imports = """\"\"\"Math head configuration.\"\"\"
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
"""

    # Fix class definition with proper docstring and fields
    fixed_content = f"""{imports}

@dataclass
class MathHeadConfig:
    \"\"\"Configuration for math reasoning head.\"\"\"

    model_dim: int = field(default=512)
    num_experts: int = field(default=8)
    expert_size: int = field(default=128)
    dropout_rate: float = field(default=0.1)
    use_bias: bool = field(default=True)
    activation: str = field(default="gelu")

    def __post_init__(self):
        \"\"\"Validate configuration after initialization.\"\"\"
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.expert_size <= 0:
            raise ValueError("expert_size must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
"""

    # Write the fixed content
    with open('src/models/reasoning/math_head_config.py', 'w') as f:
        f.write(fixed_content)

if __name__ == '__main__':
    fix_math_head_config()