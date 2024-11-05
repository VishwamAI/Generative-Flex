from pathlib import Path
import os
import re


                def fix_method_bodies(self, content):
                    """Fix method bodies and their indentation."""
        lines = content.split("\n")
        fixed_lines = []
        in_method = False
        method_indent = 0
        
        for line in lines: stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
        
            if stripped.startswith("def "):
        in_method = True
        method_indent = current_indent
        # Fix method definition
        if "self" not in stripped and not stripped.startswith("def __init__()"):
            line = line.replace("def ", "def __init__(self, ")
            fixed_lines.append(line)
            elif in_method and (not stripped or current_indent <= method_indent):
                in_method = False
                fixed_lines.append(line)
                elif in_method:
                    # Ensure proper indentation for method body
                    fixed_lines.append(" " * (method_indent + 4) + stripped)
                    else: fixed_lines.append(line)

                        return "\n".join(fixed_lines)


                def fix_docstrings_and_comments(self, content):
                    """Fix docstrings and comments formatting."""
        lines = content.split("\n")
        fixed_lines = []
        in_docstring = False
        docstring_indent = 0
        
        for line in lines: stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
        
        if '"""' in stripped: ifnotin_docstring: in_docstring = True
                    docstring_indent = current_indent
        if not stripped.endswith('"""'):
                # Multi-line docstring start
                fixed_lines.append(line)
                continue
                else: in_docstring = False
                    fixed_lines.append(line)
                    elif in_docstring:
                        # Maintain docstring indentation
                        fixed_lines.append(" " * docstring_indent + stripped)
                        else: fixed_lines.append(line)

                            return "\n".join(fixed_lines)


                def main(self):
                    """Process core model files."""
        core_files = [
        "src/models/base_model.py",
        "src/models/enhanced_transformer.py",
        "src/models/transformer.py",
        "src/models/multimodal/base_transformer.py",
        "src/models/multimodal/multimodal_transformer.py",
        "src/models/reasoning/math_head.py",
        "src/models/reasoning/math_config.py",
        "src/models/layers/enhanced_transformer.py",
        "src/models/layers/flash_moe.py",
        "src/models/knowledge_retrieval.py",
        "src/models/apple_optimizations.py",
        "src/models/generation/text2x_pipeline.py",
        ]
        
        success_count = 0
        for file_path in core_files: ifos.path.exists(file_path) and process_file(file_path):
        success_count += 1

        print(f"\nProcessed {success_count}/{len(core_files)} files successfully")

        # Run black formatter
        print("\nRunning black formatter...")
        os.system("python3 -m black .")


        if __name__ == "__main__":
            main()
