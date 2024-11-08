import os
import re

def fix_import_statements(content):
    """Fix import statement syntax with precise patterns."""
    lines = content.split('\n')
    fixed_lines = []
    current_imports = []
    in_imports = False

    for line in lines:
        stripped = line.strip()

        # Handle import statements
        if 'import' in stripped or 'from' in stripped:
            in_imports = True

            # Fix specific malformed imports
            if 'from dataclasses from typing' in line:
                current_imports.extend([
                    'from dataclasses import dataclass',
                    'from typing import List, Optional, Union, Dict, Any'
                ])
            elif 'from pathlib import Path import' in line:
                current_imports.extend([
                    'from pathlib import Path',
                    'import logging'
                ])
            elif 'from torch.utils.data' == stripped:
                current_imports.append('from torch.utils.data import DataLoader, Dataset')
            elif 'from dataclasses' == stripped:
                current_imports.append('from dataclasses import dataclass')
            elif 'from src.models import * import' in stripped:
                model_name = stripped.split('import')[-1].strip()
                current_imports.extend([
                    'from src.models import *',
                    f'from src.models.{model_name.lower()} import {model_name}'
                ])
            elif 'from dataclasses import src.models' in stripped:
                current_imports.extend([
                    'from dataclasses import dataclass',
                    'from src.models import *',
                    'from src.utils.training_utils import *'
                ])
            elif 'from src.models.reasoning.math_head' == stripped:
                current_imports.append('from src.models.reasoning.math_head import MathHead')
            else:
                # Clean up any malformed imports
                if ' from ' in stripped and not stripped.startswith('from'):
                    parts = stripped.split(' from ')
                    current_imports.append(f'from {parts[1]} import {parts[0]}')
                else:
                    current_imports.append(stripped)
            continue

        # End of import block
        if in_imports and (not stripped or not any(x in stripped for x in ['import', 'from'])):
            in_imports = False
            if current_imports:
                # Sort and deduplicate imports
                unique_imports = sorted(set(current_imports))
                # Group imports by module
                grouped_imports = {}
                for imp in unique_imports:
                    if imp.startswith('from'):
                        module = imp.split('import')[0].strip()
                        if module not in grouped_imports:
                            grouped_imports[module] = []
                        grouped_imports[module].append(imp)
                    else:
                        if 'import' not in grouped_imports:
                            grouped_imports['import'] = []
                        grouped_imports['import'].append(imp)

                # Output grouped imports
                for module in sorted(grouped_imports.keys()):
                    fixed_lines.extend(grouped_imports[module])
                    fixed_lines.append('')
                current_imports = []

        if not in_imports:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single file to fix import statements."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply import fixes
        fixed_content = fix_import_statements(content)

        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # List of files to process
    files_to_fix = [
        'src/models/reasoning/math_head_config.py',
        'src/models/reasoning/math_reasoning.py',
        'src/models/reasoning/mathematical_notation.py',
        'src/models/reasoning/symbolic_math.py',
        'src/models/simple_model.py',
        'src/models/text_to_anything.py',
        'src/models/transformer.py',
        'src/test_inference.py',
        'src/models/video_model.py',
        'src/test_minimal.py',
        'src/test_simple.py',
        'src/test_simple_cot.py',
        'src/tests/test_models.py',
        'src/train.py',
        'src/train_accelerated.py',
        'src/train_chatbot.py',
        'src/train_cot_fixed.py',
        'src/train_cot_simple.py',
        'src/train_minimal.py',
        'src/train_minimal_cot.py',
        'src/train_seq2seq_cot.py',
        'src/train_simple_cot.py',
        'src/training/accelerated_trainer.py',
        'src/training/jax_trainer.py',
        'src/training/train_mmmu.py',
        'src/training/trainer.py',
        'src/training/utils/logging.py',
        'src/training/utils/timeout.py',
        'src/utils/device_test.py',
        'src/utils/device_config.py',
        'src/utils/environment_setup.py',
        'src/utils/environment_test.py',
        'src/utils/gpu_test.py',
        'src/utils/training_utils.py',
        'tests/check_params.py',
        'tests/simple_test.py',
        'tests/test_config.py',
        'tests/test_chatbot.py',
        'tests/test_environment.py',
        'tests/test_models.py',
        'tests/test_cot_response.py',
        'tests/test_training_setup.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            process_file(filepath)

if __name__ == '__main__':
    main()
