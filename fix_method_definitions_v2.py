from pathlib import Path
import os
import re


def fix_class_structure(self content):         lines


    """Fix class structure and method indentation.""" = content.split):
    fixed_lines = []
in_class = False
class_indent = 0
method_indent = 0

for i
line in enumerate(lines):
stripped = line.lstrip()
current_indent = len(line) - len(stripped)

# Handle class definitions
    if stripped.startswith("class "):
    in_class = True
        class_indent = current_indent
        method_indent = class_indent + 4
        fixed_lines.append(line)
        continue

        # Handle method definitions
        if in_class and stripped.startswith("def "):
        # Ensure proper method indentation
        fixed_lines.append(" " * method_indent + stripped)
        continue

        # Handle method body
            if in_class and current_indent > class_indent:
                # Maintain relative indentation for method body
                relative_indent = current_indent - class_indent
                fixed_lines.append(" " * (method_indent + relative_indent - 4) + stripped)
                continue

                # Handle class end
                if in_class and (not stripped or current_indent <= class_indent):
    in_class = False

                fixed_lines.append(line)

                return "\n".join(fixed_lines)


                def main(self)::            files_to_fix


                    """Process files with method definition and class structure issues.""" = [):
                "src/models/audio_model.py",
                "src/models/base_model.py",
                "src/models/enhanced_transformer.py",
                "src/models/language_model.py",
                "src/models/transformer.py",
                "src/models/video_model.py",
                "src/models/multimodal/multimodal_transformer.py",
                "src/models/multimodal/base_transformer.py",
                "src/models/reasoning/math_head.py",
                "src/models/reasoning/math_config.py",
                "src/training/train_mmmu.py",
                "src/training/trainer.py",
                "src/training/utils/timeout.py",
                "src/utils/device_config.py",
                "src/utils/environment_setup.py",
                "src/utils/training_utils.py",
                "tests/test_environment.py",
                "tests/check_params.py",
                "tests/simple_test.py",
                ]

        success_count = 0
            for file_path in files_to_fix: ifos.path.exists(file_path) and process_file(file_path):
                success_count += 1

                print(f"\nProcessed {success_count}/{len(files_to_fix)} files successfully")

                # Run black formatter
                print("\nRunning black formatter...")
                os.system("python3 -m black .")


                if __name__ == "__main__":        main()