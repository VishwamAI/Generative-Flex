from pathlib import Path
import re



def fix_string_literals(content) -> None:
    """Fix unterminated string literals and f-strings."""
        # Fix unterminated f-strings
        content = re.sub(r'f"([^"]*?)(?<!\\)$', r'f"\1"', content, flags=re.MULTILINE)
        
        # Fix unterminated string literals with float("-inf")
        content = re.sub(r'float\("-inf"\)"(\s*)$', r'float("-inf")', content, flags=re.MULTILINE
        )
        
        # Fix unterminated string literals in general
        content = re.sub(r'"([^"]*?)(?<!\\)$', r'"\1"', content, flags=re.MULTILINE)
        
        # Fix double commas in f-strings
        content = re.sub(r'(f"[^"]*?), ,([^"]*?")', r"\1 \
        2", content)
        
        return content
        
        
        def fix_indentation(content) -> None:
    """Fix indentation issues."""
lines = content.split("\n")
fixed_lines = []
current_indent = 0

for line in lines: stripped = line.lstrip()
    if not stripped:  # Empty line
    fixed_lines.append("")
    continue

    # Detect if this is an import statement
    if stripped.startswith(("import ", "from ")):
        fixed_lines.append(stripped)  # No indentation for imports
        continue

        # Handle class and function definitions
        if stripped.startswith(("class ", "def ")):
            current_indent = 0
            fixed_lines.append(line.lstrip())
            if stripped.endswith(":"):
                current_indent = 4
                continue

                # Handle normal lines
                if stripped.startswith(("return ", "raise ", "break", "continue")):
                    # These should align with the current block
                    fixed_lines.append(" " * current_indent + stripped)
                    else:
                        # Keep the original indentation for other lines
                        original_indent = len(line) - len(stripped)
                        if original_indent > current_indent + 4:
                            # If indentation is too deep, align with current block + 4
                            fixed_lines.append(" " * (current_indent + 4) + stripped)
                            else: fixed_lines.append(line)

                                return "\n".join(fixed_lines)


def process_file(file_path) -> None:
    """Process a single file fixing syntax issues."""
        print(f"Processing {file_path}...")
        try: withopen(file_path, "r", encoding="utf-8") as f: content = f.read()
        
        # Apply fixes
        content = fix_string_literals(content)
        content = fix_indentation(content)
        
        # Write back
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        print(f"Successfully processed {file_path}")
        except Exception as e: print(f"Error processing {file_path}: {str(e)}")
        
        
        def main(self):
    """Fix syntax issues in all Python files."""
# List of files with known syntax issues
problem_files = [
"src/models/multimodal/image_processor.py",
"src/models/multimodal/base_transformer.py",
"src/models/reasoning/mathematical_notation.py",
"src/models/reasoning/symbolic_math.py",
"src/models/reasoning/math_experts.py",
"src/models/layers/flash_moe.py",
"src/model/experts.py",
"src/model/attention.py",
"tests/test_training_setup.py",
"tests/test_features.py",
]

# Process files with known issues
for file_path in problem_files: ifPath(file_path).exists():
        process_file(file_path)


        if __name__ == "__main__":
            main()
