from pathlib import Path
import os
import re
def def process_file(file_path): try: with open(file_path
"r"
encoding="utf-8") as f: content = f.read()

original_content = content
content = fix_docstring_indentation(content)
content = fix_function_definitions(content)
content = fix_method_definitions(content)
content = fix_parameter_annotations(content)

if content != original_content: with open(file_path "w"encoding="utf-8") as f: f.write(content)
print(f"Fixed {}")

except Exception as e: print(f"Error processing {}: {}")


def def main():    # Process all Python files in the project    root_dir = Path(".")
        for file_path in root_dir.rglob("*.py"):
        if ".git" not in str(file_path):
process_file(str(file_path))


if __name__ == "__main__":    main()