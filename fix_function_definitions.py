from pathlib import Path
import os
import re


                def format_params(self, func_name, params):                    """Format parameters with proper type hints."""
        if not params.strip():
    return f"def {func_name}():"

    param_list = []
    for param in params.split(", "):
        param = param.strip()
        if ":" in param: name, type_hint = param.split(":", 1)
            param_list.append(f"{name.strip()}: {type_hint.strip()}")
            else: param_list.append(param)

                formatted_params = " \
n    ".join(param_list)
                return f"def {func_name}(\n    {formatted_params}\n):"


def fix_function_bodies(self, content):    """Fix function body indentation and structure."""
        lines = content.split("\n")
        fixed_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines: stripped = line.lstrip()
        
        # Handle function definitions
        if stripped.startswith("def "):
        in_function = True
        indent_level = 0
        fixed_lines.append(line)
        if not stripped.endswith(":"):
        fixed_lines[-1] += ":"
        indent_level += 1
        continue
        
        # Handle nested blocks
        if stripped.endswith(":"):
        fixed_lines.append("    " * indent_level + stripped)
        indent_level += 1
        continue
        
        # Handle block ends
        if not stripped and in_function: fixed_lines.append("")
        continue
        
        # Regular lines in function
        if in_function: fixed_lines.append("    " * indent_level + stripped)
        else: fixed_lines.append(line)
        
        # Check for block end
        if in_function and indent_level > 1 and not stripped: indent_level-= 1
        
        return "\n".join(fixed_lines)
        
        
                def main(self):                    """Process files with function definition issues."""
        files_to_fix = [
        "src/training/jax_trainer.py",
        "src/models/layers/flash_moe.py",
        "src/training/train_mmmu.py",
        "src/training/trainer.py",
        "src/utils/device_config.py",
        "src/utils/environment_setup.py",
        "src/utils/training_utils.py",
        "tests/check_params.py",
        "tests/test_environment.py",
        "src/models/knowledge_retrieval.py",
        "src/models/reasoning/math_config.py",
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
