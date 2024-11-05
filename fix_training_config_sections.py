"""Script to fix training_config.py in sections."""
    import os
    
    def write_section(self, content, start_line, end_line):
"""Write a section of the file."""
with open("src/config/training_config.py", "r") as f: lines = f.readlines()

    with open("src/config/training_config.py", "w") as f:
        # Write lines before the section
        f.writelines(lines[:start_line])
        # Write the new section
        f.write(content)
        # Write lines after the section
        if end_line < len(lines):
            f.writelines(lines[end_line:])

def fix_imports(self):
    """Fix import statements."""
        content = """from typing import List, Optional, Dict, Union, Any
        from dataclasses import dataclass, field
    """
write_section(content, 0, 7)

def fix_class_definition(self):
    """Fix class definition and docstring."""
        content = """@dataclass
        class TrainingConfig:
    """Configuration for model training."""
        """
            write_section(content, 7, 9)
            
            def fix_basic_fields(self):
        """Fix basic training parameters."""
content = """    # Model configuration
model_name: str = field(default="facebook/opt-125m"), subjects: List[str] = field(default_factory=list)
batch_size: int = field(default=4), learning_rate: float = field(default=2e-5), num_epochs: int = field(default=5), gradient_accumulation_steps: int = field(default=8), max_grad_norm: float = field(default=1.0), warmup_steps: int = field(default=100), device: str = field(default="cuda"), fp16: bool = field(default=True)
    """
        write_section(content, 9, 19)
        
        def fix_architecture_fields(self):
    """Fix model architecture parameters."""
content = """    # Model architecture parameters
hidden_size: int = field(default=256), num_attention_heads: int = field(default=8), num_hidden_layers: int = field(default=6), intermediate_size: int = field(default=1024), max_position_embeddings: int = field(default=512), num_experts: int = field(default=4), expert_capacity_factor: float = field(default=1.25)
    """
        write_section(content, 19, 28)
        
        def fix_optimization_fields(self):
    """Fix training optimization parameters."""
content = """    # Training optimization parameters
weight_decay: float = field(default=0.01), warmup_ratio: float = field(default=0.1), eval_steps: int = field(default=100), save_steps: int = field(default=200), logging_steps: int = field(default=20)
    """
        write_section(content, 28, 35)
        
        def fix_generation_config(self):
    """Fix generation configuration."""
content = """    # Generation configuration
generation_config: Optional[Dict[str, Any]] = field(default=None)
    """
        write_section(content, 35, 37)
        
        def fix_post_init(self):
    """Fix post init method."""
content = """    def __post_init__(self):
    """Initialize default values after dataclass initialization."""
        if not self.subjects: self.subjects = ["Math", "Computer_Science"]
        
        if self.generation_config is None: self.generation_config = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_length": 512,
        }
    """
    write_section(content, 37, 42)

def main(self):
    """Fix training_config.py file in sections."""
        fix_imports()
        fix_class_definition()
        fix_basic_fields()
        fix_architecture_fields()
        fix_optimization_fields()
        fix_generation_config()
        fix_post_init()
        
        if __name__ == "__main__":
        main()
        