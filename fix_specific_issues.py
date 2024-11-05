import os
import re
    """Script to fix specific formatting issues in problematic files."""
        
        
        
        def fix_file_operations(filename) -> None:
    """Fix file operation syntax issues."""
with open(filename, "r") as f: content = f.read()

    # Add proper imports and fix file operations
    fixed = "import os\n\n" + content
    fixed = re.sub(r'with open\("([^"]+)", "r"\) as f:',
    lambda m: f'with open(os.path.join(os.path.dirname(__file__), "{m.group(1)}"), "r") as f:',
    fixed)

    with open(filename, "w") as f: f.write(fixed)


def fix_docstrings(filename) -> None:
    """Fix docstring parsing issues."""
        with open(filename, "r") as f: content = f.read()
        
        # Fix docstring formatting
        fixed = re.sub(r'"""([^"]*?)"""', lambda m: '"""\n' + m.group(1).strip() + '\n"""', content
        )
        
        with open(filename, "w") as f: f.write(fixed)
        
        
        def fix_module_syntax(filename) -> None:
    """Fix module-level syntax issues."""
with open(filename, "r") as f: content = f.read()

    # Fix module-level docstrings and imports
    if "Mixture of Experts Implementation" in content: fixed = '"""Mixture of Experts Implementation for Generative-Flex."""\n\n'
        fixed += "import torch\nimport torch.nn as nn\n\n"
        fixed += content[content.find("\n\n") + 2 :]
        elif "Flash Attention Implementation" in content: fixed = '"""Flash Attention Implementation for Generative-Flex."""\n\n'
            fixed += "import torch\nimport torch.nn as nn\n\n"
            fixed += content[content.find("\n\n") + 2 :]
            else: fixed = content

                with open(filename, "w") as f: f.write(fixed)


def main(self):
    """Fix formatting issues in specific files."""
        # Files with file operation issues
        file_op_files = [
        "fix_text_to_anything.py",
        "fix_text_to_anything_v6.py",
        "fix_text_to_anything_v7.py",
        "fix_text_to_anything_v8.py",
        "fix_string_formatting.py",
        ]
        
        # Files with docstring issues
        docstring_files = [
        "analyze_performance_by_category.py",
        "fix_flake8_comprehensive.py",
        "data/dataset_verification_utils.py",
        ]
        
        # Files with module syntax issues
        module_files = ["src/model/experts.py", "src/model/attention.py"]
        
        # Fix datasets import issue
        with open("data/verify_mapped_datasets.py", "r") as f: content = f.read()
        with open("data/verify_mapped_datasets.py", "w") as f: f.write("try:\n    from datasets import load_dataset\nexcept ImportError:\n    pass\n\n"
        + content[content.find("\n") + 1 :]
        )
        
        # Apply fixes
        for filename in file_op_files: ifos.path.exists(filename):
        print(f"Fixing file operations in {filename}")
        fix_file_operations(filename)
        
        for filename in docstring_files: ifos.path.exists(filename):
        print(f"Fixing docstrings in {filename}")
        fix_docstrings(filename)
        
        for filename in module_files: ifos.path.exists(filename):
        print(f"Fixing module syntax in {filename}")
        fix_module_syntax(filename)
        
        
        if __name__ == "__main__":
        main()
        