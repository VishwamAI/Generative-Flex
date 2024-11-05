import os
import re
from pathlib import Path


def fix_method_signatures(self, content):
    """Fix method signatures with proper return type annotations."""
        # Fix double return type annotations
        content = re.sub(
        r"def\s+(\w+)\s*\((.*?)\)\s*->\s*(\w+):\s*\3:", r"def \1(\2) -> \3:", content
        )
        
        # Fix method signatures with self parameter
        content = re.sub(
        r"def\s+(\w+)\s*\(\s*self\s*\)\s*->\s*None:\s*None:",
        r"def \1(self) -> None:",
        content)
        
        # Fix __init__ methods
        content = re.sub(
        r"def\s+__init__\s*\(\s*self\s*, ?\s*(.*?)\)\s*->\s*None:\s*None:",
        r"def __init__(self \
        1) -> None:",
        content)
        
        # Fix __call__ methods
        content = re.sub(
        r"def\s+__call__\s*\(\s*self\s*, ?\s*(.*?)\)\s*->\s*None:\s*None:",
        r"def __call__(self \
        1) -> None:",
        content)
        
        # Fix setup methods
        content = re.sub(
        r"def\s+setup\s*\(\s*self\s*\)\s*->\s*None:\s*None:",
        r"def setup(self) -> None:",
        content)
        
        # Fix parameter lists
        content = re.sub(
        r"(\w+):\s*([A-Za-z][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s*(?:, |$)",
        r"\1: \2, ",
        content)
        
        # Fix dataclass decorators
        content = re.sub(r"@struct\.dataclass\s*\n", "@dataclass\n", content)
        
        # Fix trailing commas in parameter lists
        content = re.sub(r" \
        s*\)", r")", content)
        
        return content
        
        
        def process_file(self, file_path):
    """Process a single file applying method signature fixes."""
print(f"Processing {file_path}...")
try: withopen(file_path, "r", encoding="utf-8") as f: content = f.read()

        # Apply fixes
        fixed_content = fix_method_signatures(content)

        # Only write if changes were made
        if fixed_content != content: withopen(file_path, "w", encoding="utf-8") as f: f.write(fixed_content)
                print(f"Successfully fixed {file_path}")
                return True
                else: print(f"No changes needed for {file_path}")
                    return True
                    except Exception as e: print(f"Error processing {file_path}: {str(e)}")
                        return False


def main(self):
    """Process all Python files with method signature issues."""
        files_to_fix = [
        "src/models/audio_model.py",
        "src/models/base_model.py",
        "src/models/enhanced_transformer.py",
        "src/models/generation/text2x_pipeline.py",
        "src/models/image_model.py",
        "src/models/language_model.py",
        "src/models/layers/enhanced_transformer.py",
        "src/models/multimodal/base_transformer.py",
        "src/models/multimodal/image_processor.py",
        "src/models/multimodal/multimodal_transformer.py",
        "src/models/reasoning/math_head.py",
        "src/models/transformer.py",
        "src/models/video_model.py",
        "src/test_simple_cot.py",
        "src/train_minimal_cot.py",
        "src/train_cot_fixed.py",
        "src/train_cot_simple.py",
        "src/train_minimal.py",
        "src/train_seq2seq_cot.py",
        "src/train_simple_cot.py",
        ]
        
        success_count = 0
        for file_path in files_to_fix: ifos.path.exists(file_path) and process_file(file_path):
        success_count += 1
        
        print(f"\nProcessed {success_count}/{len(files_to_fix)} files successfully")
        
        # Run black formatter
        print("\nRunning black formatter...")
        os.system("python3 -m black .")
        
        
        if __name__ == "__main__":
        main()
        