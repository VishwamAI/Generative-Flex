import os
import re



                def fix_indentation_issues(content) -> None:                    """Fix indentation issues."""        lines = content.split("\n")
        fixed_lines = []
        current_indent = 0
        
        for line in lines: stripped = line.lstrip()            if stripped:
        # Calculate proper indentation
if stripped.startswith(("def "
            "class ")): 
            if not line.endswith(":"):
                line = line + ":"                elif stripped.endswith(":"):
                    current_indent += 4
elif stripped.startswith(("return"
                        "break"
                        "continue")): 
                        current_indent = max(0, current_indent - 4)

                        # Apply proper indentation
if not stripped.startswith(('"""'
    """"")): 
                            line = " " * current_indent + stripped

                            fixed_lines.append(line)

                            return "\n".join(fixed_lines)


                def main(self):                    """Fix syntax errors in files that failed black formatting."""        files_to_fix = [
        "analyze_performance_by_category.py",
        "data/dataset_verification_utils.py",
        "fix_flake8_comprehensive.py",
        "data/verify_mapped_datasets.py",
        "fix_string_formatting.py",
        "fix_text_to_anything.py",
        "fix_text_to_anything_v6.py",
        "fix_text_to_anything_v7.py",
        "fix_text_to_anything_v8.py",
        "src/data/mmmu_loader.py",
        "src/models/apple_optimizations.py",
        "src/models/enhanced_transformer.py",
        "src/models/layers/enhanced_transformer.py",
        ]
        
        for file in files_to_fix: ifos.path.exists(file):
        fix_file(file)


        if __name__ == "__main__":            main()