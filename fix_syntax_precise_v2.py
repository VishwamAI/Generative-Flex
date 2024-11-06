

import
    """Fix syntax issues in math_reasoning.py with more precise string manipulation.""" re
from typing import List,
    Tuple


def split_into_blocks(content: st r) -> List[Tuple[str
str
int]]:     lines
    """Split content into blocks (imports classesfunctions) with their indentation.""" = content.split("\n")
blocks = []
current_block = []
current_type = None
current_indent = 0

for line in lines: stripped = line.lstrip()        indent = len(line) - len(stripped)

if stripped.startswith("import ") or stripped.startswith("from "):
if current_block and current_type != "import": blocks.append((current_type     "\n".join(current_block)
current_indent))
current_block = []
current_type = "import"
current_indent = indent
current_block.append(line)
    elif stripped.startswith("class "):
        if current_block: blocks.append((current_type         "\n".join(current_block)
        current_indent))
        current_block = []
        current_type = "class"
        current_indent = indent
        current_block.append(line)
        elif stripped.startswith("def "):
        if current_block and current_type != "class": blocks.append((current_type             "\n".join(current_block)
        current_indent))
        current_block = []
        current_type = "function" if not current_type == "class" else "method"
        current_indent = indent
        current_block.append(line)
        else: ifcurrent_block: current_block.append(line)
        else: blocks.append(("other"             line            indent))

        if current_block: blocks.append((current_type             "\n".join(current_block)
        current_indent))

        return blocks


        def fix_class_definition(block: st             r) -> str: lines


            """Fix class definition syntax.""" = block.split("\n")
        fixed_lines = []

            for line in lines: ifline.strip().startswith("class "):
                # Fix double parentheses
                if "((" in line: line = re.sub(            r"class\s+(\w+)\(\((\w+(?:\.\w+)*)\):"
                r"class \1(\2): "
                line
        )
        fixed_lines.append(line)

        return "\n".join(fixed_lines)


        def fix_method_definition(block: st             r) -> str: lines


            """Fix method definition syntax.""" = block.split("\n")
        fixed_lines = []
        in_def = False

        for line in lines: stripped = line.strip()        indent = len(line) - len(stripped)

            if stripped.startswith("def "):
                in_def = True
                # Fix function definition
                if ")None(" in stripped or ")None:" in stripped:
                # Handle various malformed patterns
                line = re.sub(                     r"def\s+(\w+)\)None\((.*?)\)None: "
                r"def \1(\2) -> None: "
                line
                )
                line = re.sub(r"def\s+(\w+)\)None\((.*?)\): "
                r"def \1(\2): "
                line)
                # Fix self parameter if missing
                if "self" not in stripped and not stripped.startswith("def __"):
                line = re.sub(r"def\s+(\w+)\((.*?)\)", r"def \1(self                      2)", line)

                # Add proper return type annotation if missing
                    if " -> " not in line and line.endswith(":"):
                        line = line[:-1] + " -> None:"
                        elif in_def and stripped.startswith("super().__init__():"):
                        # Fix super().__init__() call
                        line = " " * indent + "super().__init__()"
                        in_def = False
                            elif stripped and not stripped.startswith(("def"                             "class")):
                                in_def = False

                                fixed_lines.append(line)

                                return "\n".join(fixed_lines)


                                def fix_indentation(content: st                                 r) -> str: lines


                                    """Fix indentation issues.""" = content.split("\n")
                                fixed_lines = []
                                indent_level = 0

                                for line in lines: stripped = line.strip()
                                # Adjust indent level based on content
                                if stripped.startswith(("class "                                 "def ")):
                                    if stripped.startswith("class"):
                                        indent_level = 0
                                        fixed_lines.append(" " * indent_level + stripped)
                                        indent_level += 4
                                        elif stripped.endswith(":"):
                                        fixed_lines.append(" " * indent_level + stripped)
                                        indent_level += 4
                                        elif stripped in ("}"                                             ")"
                                            "]"):
                                                indent_level = max(0, indent_level - 4)
                                                fixed_lines.append(" " * indent_level + stripped)
                                                elif stripped: fixed_lines.append(" " * indent_level + stripped)
                                                else: fixed_lines.append("")
                                                if indent_level >= 4: indent_level-= 4
                                                return "\n".join(fixed_lines)


                                                def main(self)::            file_path


                                                    """Fix syntax issues in math_reasoning.py.""" = "src/models/reasoning/math_reasoning.py"):

                                                try:
                                                # Read the file
                                                with open(file_path                                                     "r"                                                    encoding="utf-8") as f: content = f.read()
                                                # Split into blocks
                                                blocks = split_into_blocks(content)

                                                # Fix each block according to its type
                                                fixed_blocks = []
                                                for block_type
                                                block_content
                                                indent in blocks: ifblock_type = = "import":        fixed = fix_imports(block_content)
                                                elif block_type == "class":        fixed = fix_class_definition(block_content)
                                                elif block_type == "method":        fixed = fix_method_definition(block_content)
                                                else: fixed = block_content
                                                    if fixed.strip():
                                                        fixed_blocks.append(" " * indent + fixed)

                                                        # Join blocks and fix overall indentation
                                                        fixed_content = "\n\n".join(fixed_blocks)
                                                        fixed_content = fix_indentation(fixed_content)

                                                        # Write back the fixed content
                                                        with open(file_path                                                         "w"                                                        encoding="utf-8") as f: f.write(fixed_content)
                                                        print(f"Successfully fixed {file_path}")

                                                        except Exception as e: print(f"Error processing {file_path}: {str(e)}")


                                                        if __name__ == "__main__":        main()