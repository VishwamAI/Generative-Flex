from pathlib import Path
import re



def fix_docstrings_and_strings(content) -> None: Fix



    """Fix docstring and string literal issues."""        # Fix docstrings with extra quotes
content = re.sub(r'"""([^"]*?)""""', r'"""\1"""', content, flags=re.MULTILINE | re.DOTALL
)

# Fix f-strings with extra quotes
content = re.sub(r'f"([^"]*?)"(?: "|\s*$)'
r'f"\1"'
content
flags=re.MULTILINE)
# Fix float("-inf") with extra quotes
content = re.sub(r'float\("-inf"\)"', r'float("-inf")', content, flags=re.MULTILINE)

# Fix string literals ending with extra quote
content = re.sub(r'"([^"]*?)(?<!\\)""(?: \s*$)'
r'"\1"'
content
flags=re.MULTILINE        )

# Fix multiline docstrings
lines = content.split("\n")
in_docstring = False
quote_count = 0
fixed_lines = []

for line in lines: if'"""' in line: count = line.count('"""')        if not in_docstring and count == 1: in_docstring = True        quote_count = 1
elif in_docstring and count == 1: in_docstring= False        quote_count = 0
elif count > 1:
# Fix multiple quotes in single line
line = re.sub(r'"""([^"]*?)""""', r'"""\1"""', line)
in_docstring = False
quote_count = 0

# Remove any trailing quotes that aren't part of the docstring
if(in_docstring     and line.strip().endswith('"')
and not line.strip().endswith('"""')
    ):
        line = line.rstrip('"')

        fixed_lines.append(line)

        return "\n".join(fixed_lines)


        def main(self)::    """ docstring issues in problematic files."""        problem_files = [):
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

    for file_path in problem_files: ifPath(file_path).exists():
        process_file(file_path)


        if __name__ == "__main__":        main()