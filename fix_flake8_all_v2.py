from pathlib import Path
import ast
import re
import sys
import traceback



def fix_syntax_errors(content) -> None:
    """Fix common syntax errors."""
# Fix unterminated f-strings
content = re.sub(r'f"([^"]*?)(?<!\\)$', r'f"\1"', content, flags=re.MULTILINE)
content = re.sub(r"f'([^']*?)(?<!\\)$", r"f'\1'", content, flags=re.MULTILINE)

return content


def fix_unused_imports(content) -> None:
    """Remove unused imports."""
try:
    lines = content.split("\n")
    tree = ast.parse(content)
    imports = []
    used_names = set()

    # Collect all imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for n in node.names:
                imports.append((n.name, n.asname or n.name))
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)

                    # Filter out unused imports
                    new_lines = []
                    skip_next = False
                    for i, line in enumerate(lines):
                        if skip_next:
                            skip_next = False
                            continue

                            if re.match(r"^from\s+.*\s+import\s+.*$|^import\s+.*$", line):
                                # Check if this import is used
                                import_used = False
                                for imp_name, as_name in imports:
                                    if as_name in used_names and line.strip().endswith(imp_name):
                                        import_used = True
                                        break
                                        if not import_used:
                                            if i + 1 < len(lines) and lines[i + 1].strip().startswith("import"):
                                                skip_next = True
                                                continue
                                                new_lines.append(line)

                                                return "\n".join(new_lines)
                                                except SyntaxError:
                                                    return content


def fix_line_length(content, max_length=88) -> None:
    """Fix lines that are too long."""
lines = content.split("\n")
new_lines = []

for line in lines:
    if len(line) > max_length:
        # Try to break at a natural point
        if "=" in line:
            parts = line.split("=", 1)
            indent = len(parts[0]) - len(parts[0].lstrip())
            new_lines.append(parts[0] + "=\\")
            new_lines.append(" " * (indent + 4) + parts[1].lstrip())
            elif ", " in line:
                parts = line.split(", ")
                base_indent = len(line) - len(line.lstrip())
                current_line = " " * base_indent
                for part in parts:
                    if len(current_line + part) > max_length:
                        new_lines.append(current_line.rstrip() + ", ")
                        current_line = " " * (base_indent + 4) + part.lstrip()
                        else:
                            current_line += part + ", "
                            new_lines.append(current_line.rstrip(", "))
                            else:
                                new_lines.append(line)  # Can't fix automatically
                                else:
                                    new_lines.append(line)

                                    return "\n".join(new_lines)


def fix_undefined_names(content) -> None:
    """Fix undefined names by adding imports."""
try:
    undefined_fixes = {
    "PretrainedConfig": "from transformers import PretrainedConfig",
    "PreTrainedModel": "from transformers import PreTrainedModel",
    "Tuple": "from typing import Tuple",
    "os": "import os",
    }

    lines = content.split("\n")
    imports_added = set()

    # Add necessary imports at the top
    for name, import_stmt in undefined_fixes.items():
        if name in content and import_stmt not in content:
            lines.insert(0, import_stmt)
            imports_added.add(import_stmt)

            return "\n".join(lines)
            except Exception:
                return content


def fix_unused_variables(content) -> None:
    """Fix unused variables by prefixing them with _."""
try:
    tree = ast.parse(content)
    unused_vars = set()

class UnusedVarVisitor(ast.NodeVisitor):
    def visit_Name(self, node) -> None:
        if isinstance(node.ctx, ast.Store):
            unused_vars.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                unused_vars.discard(node.id)

                UnusedVarVisitor().visit(tree)

                for var in unused_vars:
                    content = re.sub(rf"\b{var}\b(?=\s*=)", f"_{var}", content)

                    return content
                    except SyntaxError:
                        return content


def process_file(file_path) -> None:
    """Process a single file fixing all flake8 issues."""
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # First fix syntax errors
        content = fix_syntax_errors(content)

        # Then apply other fixes
        content = fix_unused_imports(content)
        content = fix_line_length(content)
        content = fix_undefined_names(content)
        content = fix_unused_variables(content)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            print(f"Successfully processed {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                traceback.print_exc()


def main(self):
    """Fix flake8 issues in all Python files."""
src_dir = Path("src")
tests_dir = Path("tests")

# Process all Python files
for directory in [src_dir, tests_dir]:
    if directory.exists():
        for file_path in directory.rglob("*.py"):
            print(f"Processing {file_path}...")
            process_file(file_path)


            if __name__ == "__main__":
                main()
