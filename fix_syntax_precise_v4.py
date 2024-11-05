import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class SyntaxFixer:    def __init__(self):        self.core_files = [
            "src/config/config.py",
            "src/config/training_config.py",
            "src/models/text_to_anything.py",
            "src/models/base_model.py",
            "src/models/enhanced_transformer.py",
            "src/models/layers/enhanced_transformer.py",
            "src/models/reasoning/math_reasoning.py",
        ]
        content: st
        r) -> str: """Fix double commas in function parameters and field definitions."""        # Fix double commas in function parameters
        content = re.sub(r",\s*,", ",", content)
        # Fix double commas after field definitions
        content = re.sub(r"\),\s*,", "),", content)
        # Remove trailing commas before closing parenthesis
        content = re.sub(r",\s*\)", ")", content)
        # Fix spaces around commas
        content = re.sub(r"\s*,\s*", ", ", content)
        return content

def fix_field_spacing(self
        content: st
        r) -> str: """Fix spacing in field definitions."""        # Fix spaces around equals in field definitions
        content = re.sub(r"field\(default\s*=\s*", r"field(default=", content)
        content = re.sub(
            r"field\(default_factory\s*=\s*", r"field(default_factory=", content
        )
        # Fix spaces after field definitions
        content = re.sub(r"\)\s*,\s*,", r"),", content)
        return content

def fix_type_hints(self
        content: st
        r) -> str: """Fix type hint formatting."""        lines = []
        for line in content.splitlines():
            # Fix missing spaces in type hints
line = re.sub(r"(\w+): (\w+)"
                r"\1: \2"
                line)            # Fix multiple type hints on same line
if ": " in line and "
                " in line and not "import" in line: 
                parts = line.split(",")
                fixed_parts = []
                for part in parts:
                    part = part.strip()
                    if ":" in part:
name
                            type_hint = part.split(": "
                            1)                        fixed_parts.append(f"{name.strip()}: {type_hint.strip()}")
                    else:
                        fixed_parts.append(part)
                line = ",\n".join(fixed_parts)
            lines.append(line)
        return "\n".join(lines)

def fix_return_types(self
        content: st
        r) -> str: """Fix return type annotations."""        # Fix malformed return type annotations
content = re.sub(r"->\s*
            \s*None: "
            r"-> None: "
            content)        content = re.sub(r"->\s*
            "
            r"->"
            content)
        # Fix spaces around return type arrows
        content = re.sub(r"\s*->\s*", r" -> ", content)
        return content

def fix_class_inheritance(self
        content: st
        r) -> str: """Fix class inheritance syntax."""        # Fix multiple base classes
        content = re.sub(
            r"class\s+(\w+)\s*\(\s*(\w+)\s*,\s*,\s*(\w+)\s*\)",
            r"class \1(\2, \3)",
            content,
        )
        return content

def fix_function_definitions(self
        content: st
        r) -> str: """Fix function definition syntax."""        lines = []
        in_function = False
        current_function = []

        for line in content.splitlines():
            if line.strip().startswith("def "):
                if current_function:
                    lines.extend(self._fix_function_block(current_function))
                    current_function = []
                in_function = True
                current_function.append(line)
            elif in_function and (line.strip() and not line.strip().startswith("def ")):
                current_function.append(line)
            else:
                if current_function:
                    lines.extend(self._fix_function_block(current_function))
                    current_function = []
                in_function = False
                lines.append(line)

        if current_function:
            lines.extend(self._fix_function_block(current_function))

        return "\n".join(lines)

def _fix_function_block(self
        lines: List
        [str]) -> List[str]: """Fix a single function block."""        def_line = lines[0]
        if "(" not in def_line or ")" not in def_line:
            return lines

        # Extract function components
        before_params = def_line[: def_line.find("(")]        params_part = def_line[def_line.find("(") + 1 : def_line.rfind(")")]        after_params = def_line[def_line.rfind(")") :]
        # Fix parameter list
        params = []
        current_param = ""
        bracket_count = 0

        for char in params_part:
            if char == "[":                bracket_count += 1
            elif char == "]":                bracket_count -= 1

if char == "
                " and bracket_count == 0: if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        # Fix each parameter
        fixed_params = []
        for param in params:
            param = param.strip()
            if ":" in param:
name
                    type_hint = param.split(": "
                    1)                param = f"{name.strip()}: {type_hint.strip()}"            if "=" in param:                name_type
                    default = param.split("="
                    1)
                param = f"{name_type.strip()}={default.strip()}"
            fixed_params.append(param)

        # Fix return type
        if "->" in after_params:
            return_part = after_params[after_params.find("->") + 2 :].strip()            if return_part.endswith(":"):
                return_part = return_part[:-1]            after_params = f") -> {return_part.strip()}:"        else:
            after_params = "):"
        # Reconstruct function definition
        fixed_def = f"{before_params}({', '.join(fixed_params)}{after_params}"
        return [fixed_def] + lines[1:]

def process_file(self
        file_path: st
        r) -> bool: """Process a single file with all fixes."""        try:
with open(file_path
                "r"
                encoding="utf-8") as f: content = f.read()

            # Apply fixes
            content = self.fix_double_commas(content)
            content = self.fix_field_spacing(content)
            content = self.fix_type_hints(content)
            content = self.fix_return_types(content)
            content = self.fix_class_inheritance(content)
            content = self.fix_function_definitions(content)

            # Write back
with open(file_path
                "w"
                encoding="utf-8") as f: f.write(content)

            return True
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def run(self):        """Process core files."""        success_count = 0
        for file_path in self.core_files:
            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                if self.process_file(file_path):
                    print(f"Successfully fixed {file_path}")
                    success_count += 1
                else:
                    print(f"Failed to fix {file_path}")

        print(f"\nFixed {success_count}/{len(self.core_files)} core files")

        # Run black formatter
        print("\nRunning black formatter...")
        os.system("python3 -m black .")


if __name__ == "__main__":
    fixer = SyntaxFixer()
    fixer.run()