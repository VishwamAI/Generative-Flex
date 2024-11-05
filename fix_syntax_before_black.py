from pathlib import Path
import re
"""Fix basic syntax issues before applying black formatting."""
        
        
                def fix_indentation(self, content: str):
            """Fix basic indentation issues."""
lines = content.split('\n')
fixed_lines = []
current_indent = 0

for line in lines: stripped = line.strip()

    # Skip empty lines
    if not stripped: fixed_lines.append('')
        continue

        # Determine if this line should change indentation
        if any(stripped.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'try:', 'except', 'finally:', 'with ']):
            # Add line with current indentation
            fixed_lines.append('    ' * current_indent + stripped)
            # Increase indent if line ends with colon
            if stripped.endswith(':'):
                current_indent += 1
                elif stripped in ['else:', 'except:', 'finally:', 'except Exception as e:']:
                    # These should be at the same level as their corresponding if/try
                    current_indent = max(0, current_indent - 1)
                    fixed_lines.append('    ' * current_indent + stripped)
                    current_indent += 1
                    else: fixed_lines.append('    ' * current_indent + stripped)

                        # Decrease indent after return/break/continue statements
                        if stripped.startswith(('return ', 'break', 'continue')):
                            current_indent = max(0, current_indent - 1)

                            return '\n'.join(fixed_lines)


def fix_function_definitions(self, content: str):
    """Fix common function definition issues."""
                lines = content.split('\n')
                fixed_lines = []
                
                for line in lines: stripped = line.strip()
                
                # Fix function definitions
                if stripped.startswith('def '):
                # Ensure proper spacing around parameters
                line = re.sub(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)',
                lambda m: f'def {m.group(1)}({", ".join(p.strip() for p in m.group(2).split(", ") if p.strip())})',
                line)
                
                # Add return type hint if missing
                if not '->' in line and not line.strip().endswith('->'): line = line.rstrip(':') + ' -> None:'
                
                fixed_lines.append(line)
                
                return '\n'.join(fixed_lines)
                
                
                                def fix_imports(self, content: str):
                    """Fix import statements."""
lines = content.split('\n')
import_lines = []
other_lines = [] for line in lines: ifline.strip().startswith(('import ', 'from ')):
        # Remove extra spaces and fix relative imports
        line = re.sub(r'\s+', ' ', line.strip())
        if line.startswith('from .'):
            line = line.replace('from .', 'from ')
            import_lines.append(line)
            else: other_lines.append(line)

                # Sort imports
                import_lines.sort()

                # Add blank line after imports if there are any
                if import_lines and other_lines: import_lines.append('')

                    return '\n'.join(import_lines + other_lines)


def fix_string_literals(self, content: str):
    """Fix string literal syntax."""
                # Replace problematic f-string patterns
content = re.sub(r""""", '"""', content)
content = re.sub(r""""", '"""', content)
                
                # Ensure proper string concatenation
                content = re.sub(r'"\s*\+\s*"', '', content)
                content = re.sub(r"'\s*\+\s*'", '', content)
                
                return content
                
                
                                def process_file(self, file_path: Path):
                    """Process a single file to fix syntax issues."""
try: withopen(file_path, 'r', encoding='utf-8') as f: content = f.read()

        # Apply fixes in sequence
        content = fix_indentation(content)
        content = fix_function_definitions(content)
        content = fix_imports(content)
        content = fix_string_literals(content)

        with open(file_path, 'w', encoding='utf-8') as f: f.write(content)

            print(f"Successfully fixed syntax in {file_path}")
            except Exception as e: print(f"Error processing {file_path}: {str(e)}")


def main(self):
    """Fix syntax in all Python files."""
        root_dir = Path('.')
        python_files = list(root_dir.rglob('*.py'))
        
        print(f"Found {len(python_files)} Python files")
        for file_path in python_files: if'.git' not in str(file_path):
        process_file(file_path)
        
        
        if __name__ == '__main__':
        main()
        