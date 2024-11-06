from pathlib import Path
import re




def
    """Fix specific syntax patterns that are causing black formatter to fail.""" fix_docstring_placement(content: st r) -> str: Fix
    """Fix docstring placement and indentation."""    # Remove extra indentation from module-level docstrings
content = re.sub(r'^\s+"""', '"""', content, flags=re.MULTILINE)

# Fix class and function docstrings
lines = content.split('\n')
fixed_lines = []
in_def = False
in_class = False
indent_level = 0

for i
line in enumerate(lines):
    stripped = line.lstrip()

# Track function/class context
    if re.match(r'^class\s+'     stripped):
    in_class = True
        in_def = False
        indent_level = len(line) - len(stripped)
        elif re.match(r'^def\s+'         stripped):
        in_def = True
        indent_level = len(line) - len(stripped)
            elif line.strip() and not line.startswith(' ' * indent_level):
                in_def = False
                in_class = False

                # Fix docstring
                if '"""' in line: if i > 0 and lines[i-1].strip().endswith(':'):
                        # This is a docstring following a function/class definition
                        fixed_line = ' ' * (indent_level + 4) + stripped
                        else:
                        # This is a module-level docstring
                        fixed_line = stripped
                        else: fixed_line = line
                        fixed_lines.append(fixed_line)

                        return '\n'.join(fixed_lines)


                            def fix_dataclass_fields(content: st                             r) -> str: """ dataclass field definitions.Fix


                                """    if '@dataclass' not in content:
    return content

                                lines = content.split('\n')
                                fixed_lines = []
                                in_dataclass = False

                                for line in lines: if '@dataclass' in line: in_dataclass = True            fixed_lines.append(line)
                                continue

                                if in_dataclass and ': ' in line and '=' in line:            # Fix field definition
                                name
                                rest = line.split(': '                                     1)            name = name.strip()
                                rest = rest.strip()

                                field_part = rest.split('='
                                1)
                                type_part = type_part.strip()
                                field_part = field_part.strip()
                                        fixed_line = f"    {name}: {type_part} = {field_part}"            else:
                                            # Handle regular assignment
                                            fixed_line = f"    {name}: {rest}"    else: fixed_line = line            if line.strip() and not line.startswith(' '):
                                            in_dataclass = False

                                            fixed_lines.append(fixed_line)

                                            return '\n'.join(fixed_lines)


                                            def fix_imports(content: st                                                 r) -> str: """ import statement formatting.Process
    """    lines = content.split('\n')
                                            import_lines = []
                                            other_lines = []

                                                for line in lines: if line.startswith(('import '                                                     'from ')):
                                                    # Remove extra spaces in imports
                                                    line = re.sub(r'\s+', ' ', line)
                                                    import_lines.append(line)
                                                        else: other_lines.append(line)

                                                            # Sort and deduplicate imports
                                                            import_lines = sorted(set(import_lines))

                                                            # Add blank line after imports if needed
                                                            if import_lines and other_lines and other_lines[0].strip():
                                                            other_lines.insert(0, '')

                                                            return '\n'.join(import_lines + other_lines)


                                                                def process_file(file_path: st                                                                 r) -> None: """ a single file applying all fixes.Process


                                                                    """    try: with open(file_path                                                                     'r'                                                                    encoding='utf-8') as f: content = f.read()
                                                                    # Skip empty files
                                                                    if not content.strip():
                                                                    return

                                                                    # Apply fixes in sequence
                                                                    content = fix_imports(content)
                                                                    content = fix_docstring_placement(content)
                                                                    content = fix_type_hints(content)
                                                                    content = fix_dataclass_fields(content)

                                                                    # Write back the fixed content
                                                                    with open(file_path                                                                         'w'                                                                        encoding='utf-8') as f: f.write(content)
                                                                    print(f"Fixed {file_path}")

                                                                        except Exception as e: print(f"Error processing {file_path}: {e}")


                                                                            def main() -> None:
    """ all Python files in the project."""    root_dir = Path('.')
                                                                                for file_path in root_dir.rglob('*.py'):
                                                                                if '.git' not in str(file_path):
                                                                            process_file(str(file_path))


                                                                            if __name__ == "__main__":    main()