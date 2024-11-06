import os
import re

def fix_imports(content):
    # Fix trailing commas in imports and consolidate multi-line imports
    def fix_import(match):
        import_stmt = match.group(0).strip()
        if import_stmt.endswith(','):
            import_stmt = import_stmt[:-1]
        return import_stmt

    content = re.sub(r'from\s+[\w.]+\s+import\s+[\w\s,]+(?:,\s*$|\n\s*$)',
                    fix_import,
                    content,
                    flags=re.MULTILINE)
    return content

def fix_class_init(content):
    # Fix class initialization and self assignments
    def fix_init(match):
        indent = match.group(1)
        var_name = match.group(2)
        value = match.group(3)
        return f"{}self.{} = {}"

    content = re.sub(r'(\s*)self\s*\n([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^\n]+)',
                    fix_init,
                    content)
    return content

def fix_docstrings(content):
    # Fix docstring placement and format
    def fix_docstring(match):
        indent = match.group(1)
        func_def = match.group(2)
        docstring = match.group(3)

        # Clean up docstring content
        docstring_lines = docstring.strip().split('\n')
        cleaned_lines = []
        for line in docstring_lines:
            line = line.strip()
            if line.startswith('"""') and line.endswith('"""'):
                line = line[3:-3].strip()
            if line:
                cleaned_lines.append(line)

        # Format docstring
        if cleaned_lines:
            formatted_docstring = f'{}    """\n'
            for line in cleaned_lines:
                formatted_docstring += f'{}    {}\n'
            formatted_docstring += f'{}    """'
            return f"{}def {}:\n{}"
        return f"{}def {}:"

    content = re.sub(r'(\s*)def\s+([^\n:]+):\s*\n\s*"""[^"]*"""',
                    fix_docstring,
                    content)
    return content

def fix_dict_creation(content):
    # Fix dictionary creation and formatting
    def fix_dict(match):
        indent = match.group(1)
        content = match.group(2)

        # Clean up dictionary content
        content = content.strip()
        if not content:
            return f"{}{}}"

        # Format dictionary entries
        entries = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('}'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    entries.append(f'{}: {}')
                elif '**' in line:
                    entries.append(line)

        if entries:
            return f"{}{}    " + f",\n{}    ".join(entries) + f"\n{}}}"
        return f"{}{}}"

    content = re.sub(r'(\s*){}]*)}',
                    fix_dict,
                    content,
                    flags=re.DOTALL)
    return content

def fix_file_operations(content):
    # Fix file operation syntax
    content = re.sub(r'open\(([^,]+)\s+"([^"]+)"\)',
                    r'open(\1, "\2")',
                    content)
    return content

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_imports(content)
        content = fix_class_init(content)
        content = fix_docstrings(content)
        content = fix_dict_creation(content)
        content = fix_file_operations(content)

        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Successfully processed {}")
        else:
            print(f"No changes needed for {}")

    except Exception as e:
        print(f"Error processing {}: {}")

def main():
    # Process all Python files
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == '__main__':
    main()
