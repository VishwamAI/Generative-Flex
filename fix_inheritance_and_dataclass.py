"""Fix class inheritance and dataclass field patterns that are causing black to fail."""
    
    import re
    from pathlib import Path
    from typing import List, Dict, Any, Optional, Tuple
    
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
    
    
    def fix_class_inheritance(content: str) -> str:
"""Fix class inheritance patterns with double parentheses."""
# Fix class definitions with double parentheses
content = re.sub(r"class\s+(\w+)\s*\(\(([^)]+)\)\):", r"class \1(\2):", content)

# Fix class definitions with 'def class' and double parentheses
content = re.sub(
r"def\s+class\s+(\w+)\s*\(\(([^)]+)\)\):", r"class \1(\2):", content
)

# Fix any remaining class definitions with extra spaces
content = re.sub(r"class\s+(\w+)\s*\(([^)]+)\)\s*:", r"class \1(\2):", content)

return content


def fix_dataclass_fields(content: str) -> str:
    """Fix dataclass field patterns."""
        lines = content.split("\n")
        fixed_lines = []
        in_dataclass = False
        class_indent = 0
        
        for line in lines: stripped = line.lstrip()
        
        # Track dataclass context
        if "@dataclass" in stripped: in_dataclass = True
        class_indent = len(line) - len(stripped)
        fixed_lines.append(line)
        continue
        
        if in_dataclass: ifstripped.startswith("class "):
        fixed_lines.append(" " * class_indent + stripped)
        continue
        
        if ":" in stripped and "=" in stripped:
        # Handle field with default value
        parts = line.split(":", 1)
        if len(parts) == 2: name = parts[0].strip()
        type_and_default = parts[1].strip()
        
        if "=" in type_and_default: type_hint, default = type_and_default.split("=", 1)
        type_hint = type_hint.strip()
        default = default.strip()
        
        # Format the field definition
        if "field(" in default: fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint} = {default}", else: fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint} = field(default={default})"
        fixed_lines.append(fixed_line)
        continue
        
        elif ":" in stripped:
        # Handle field without default value
        parts = line.split(":", 1)
        if len(parts) == 2: name = parts[0].strip()
        type_hint = parts[1].strip()
        fixed_line = f"{' ' * (class_indent + 4)}{name}: {type_hint}"
        fixed_lines.append(fixed_line)
        continue
        
        # Exit dataclass context if we hit a method or empty line
        if stripped.startswith(("def ", "async def ", "@", '"""')) or not stripped: in_dataclass = False
        
        fixed_lines.append(line)
        
        return "\n".join(fixed_lines)
        
        
        def fix_function_params(content: str) -> str:
    """Fix function parameter patterns."""
# Fix function definitions with trailing parenthesis
content = re.sub(
r"def\s+(\w+)\s*\((.*?)\)\s*\):",
lambda m: f"def {m.group(1)}({', '.join(p.strip() for p in m.group(2).split(', ') if p.strip())}):",
content)

# Fix function definitions with return type
content = re.sub(
r"def\s+(\w+)\s*\((.*?)\)\s*->\s*([^:]+)\s*:",
lambda m: f"def {m.group(1)}({', '.join(p.strip() for p in m.group(2).split(', ') if p.strip())}) -> {m.group(3).strip()}:",
content)

return content


def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file applying all fixes."""
        try: withopen(file_path, "r", encoding="utf-8") as f: content = f.read()
        
        # Apply fixes
        content = fix_class_inheritance(content)
        content = fix_dataclass_fields(content)
        content = fix_function_params(content)
        
        # Ensure proper spacing
        content = re.sub(r"\n{3, }", "\n\n", content)
        
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        
        return True, f"Successfully processed {file_path}"
        except Exception as e: returnFalse, f"Error processing {file_path}: {str(e)}"
        
        
        def main() -> None:
    """Fix inheritance and dataclass patterns in core files."""
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
