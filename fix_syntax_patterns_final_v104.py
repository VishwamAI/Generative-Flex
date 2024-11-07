import os
import re

def remove_all_docstrings_and_comments(content: str) -> str:
    """Remove all docstrings and comments from the content."""
    # Remove all docstrings
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    # Remove all comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    return content

def add_minimal_module_docstring(content: str) -> str:
    """Add minimal module-level docstring."""
    lines = content.split('\n')
    # Add minimal module docstring at the start
    result = ['"""."""', '']
    # Skip any empty lines at the start
    start_idx = 0
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    result.extend(lines[start_idx:])
    return '\n'.join(result)

def fix_class_and_method_definitions(content: str) -> str:
    """Fix class and method definitions."""
    lines = []
    for line in content.split('\n'):
        if line.strip():
            # Fix class definitions
            if line.strip().startswith('class ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            # Fix method definitions
            elif line.strip().startswith('def ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            # Fix dataclass syntax
            elif '@dataclass' in line and 'class:' in line:
                line = line.replace('class:', 'class')
        lines.append(line)
    return '\n'.join(lines)

def fix_imports(content: str) -> str:
    """Fix import statements."""
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('from'):
            # Fix double from statements
            line = re.sub(r'from\s+\w+\s+from\s+', 'from ', line)
            # Fix multiple imports
            if ',' in line:
                base = line.split('import')[0].strip()
                imports = [imp.strip() for imp in line.split('import')[1].split(',')]
                for imp in imports:
                    lines.append(f"{base} import {imp}")
                continue
        lines.append(line)
    return '\n'.join(lines)

def fix_indentation(content: str) -> str:
    """Fix indentation issues."""
    lines = []
    current_indent = 0
    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped:
            lines.append('')
            continue

        # Decrease indent for these keywords
        if stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ', 'elif:')):
            current_indent = max(0, current_indent - 4)

        # Add proper indentation
        if stripped:
            lines.append(' ' * current_indent + stripped)

        # Increase indent after these patterns
        if stripped.endswith(':'):
            current_indent += 4

        # Decrease indent after these keywords
        if stripped.startswith(('return', 'break', 'continue', 'raise', 'pass')):
            current_indent = max(0, current_indent - 4)

    return '\n'.join(lines)

def process_file(filepath: str) -> None:
    """Process a single file."""
    print(f"Processing {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in sequence
        content = remove_all_docstrings_and_comments(content)
        content = add_minimal_module_docstring(content)
        content = fix_class_and_method_definitions(content)
        content = fix_imports(content)
        content = fix_indentation(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Process files with syntax issues."""
    files_to_process = [
        'src/models/reasoning/math_experts.py',
        'src/models/reasoning/math_head.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_config.py',
        'src/utils/device_test.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_environment.py',
        'tests/test_features.py',
        'tests/test_models.py',
        'src/training/train_mmmu.py',
        'src/training/jax_trainer.py',
        'src/training/trainer.py',
        'src/models/text_to_anything.py',
        'src/models/apple_optimizations.py',
        'src/models/knowledge_retrieval.py',
        'src/models/multimodal/base_transformer.py',
        'src/models/multimodal/multimodal_transformer.py',
        'src/models/multimodal/image_processor.py',
        'src/models/layers/enhanced_transformer.py',
        'src/models/layers/flash_moe.py',
        'src/data/mmmu_dataloader.py',
        'src/data/math_tokenizer.py',
        'src/config/config.py',
        'src/config/training_config.py'
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
