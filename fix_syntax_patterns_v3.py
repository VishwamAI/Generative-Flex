"""Fix specific syntax patterns that are causing black formatter to fail."""
    
    import re
    from pathlib import Path
    from typing import List, Dict, Any, Optional, Tuple, Set
    
    CORE_FILES = [
    "src/models/text_to_anything.py",
    "src/models/reasoning/math_reasoning.py",
    "src/training/jax_trainer.py",
    "src/config/training_config.py",
    "src/data/math_tokenizer.py",
    "tests/test_models.py",
    "tests/test_features.py",
    "src/models/apple_optimizations.py",
    "src/data/mmmu_dataloader.py",
    "src/config/config.py",
    ]
    
    
def ensure_imports(content: st, r) -> str:    """Ensure necessary imports are present at the top."""
        required_imports = {
        "from dataclasses import dataclass, field",
        "from typing import Optional, Union, List, Dict, Any, Tuple",
        "import unittest",
        "import torch.nn as nn",
        "from flax.training import train_state",
        "from transformers import PreTrainedTokenizer",
        }
        
        # Check which imports are needed
        needed_imports = set()
        if "field(" in content: needed_imports.add("from dataclasses import dataclass, field")
        if "@dataclass" in content: needed_imports.add("from dataclasses import dataclass, field")
        if "unittest.TestCase" in content: needed_imports.add("import unittest")
        if "nn.Module" in content: needed_imports.add("import torch.nn as nn")
        if "train_state.TrainState" in content: needed_imports.add("from flax.training import train_state")
        if "PreTrainedTokenizer" in content: needed_imports.add("from transformers import PreTrainedTokenizer")
        if any(
        type_hint in content
        for type_hint in ["Optional", "Union", "List", "Dict", "Any", "Tuple"]
        ):
        needed_imports.add("from typing import Optional, Union, List, Dict, Any, Tuple")
        
        # Get existing imports
        existing_imports = set()
        for line in content.split("\n"):
        if line.strip().startswith(("import ", "from ")):
        existing_imports.add(line.strip())
        
        # Add missing imports at the top
        new_imports = needed_imports - existing_imports
    if new_imports: import_block = "\n".join(sorted(new_imports))
if content.startswith('"""'):
    docstring_end = content.find('"""', 3) + 3
        content = (
        content[:docstring_end]
        + "\n\n"
        + import_block
        + "\n"
        + content[docstring_end:]
        )
        else: content = import_block + "\n\n" + content
        
        return content
        
        
def main() -> None:    """Fix syntax patterns in core files."""
        print("Starting to process core files...")
        successful = 0
        failed = 0
        
        for file_path in CORE_FILES: ifPath(file_path).exists():
        print(f"\nProcessing {file_path}")
        success, message = process_file(file_path)
        print(message)
        if success: successful+= 1
        else: failed+= 1
        
        print(
        f"\nProcessing complete: {successful} files successful, {failed} files failed"
        )
        
        
        if __name__ == "__main__":
        main()
        