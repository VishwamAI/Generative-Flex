import os
import re

def fix_imports(content):
    # Fix trailing commas in imports
    content = re.sub(r'from\s+[\w.]+\s+import\s+[\w\s,]+,\s*$',
                    lambda m: m.group().rstrip(','),
                    content,
                    flags=re.MULTILINE)
    return content

def fix_docstrings(content):
    # Fix docstring placement and format
    content = re.sub(r'(\s*)"""[^"]*"""\s*\.\s*', r'\1', content)  # Remove malformed docstrings
    content = re.sub(r'(\s*)def\s+([^\n(]+)\([^)]*\):\s*\n\s*([^"\n]+)\s*"""',
                    r'\1def \2():\n\1    """\n\1    \3\n\1    """',
                    content)
    return content

def fix_method_definitions(content):
    # Fix method definitions and parameters
    def fix_method(match):
        indent = match.group(1)
        name = match.group(2)
        params = match.group(3)
        # Clean up parameters
        params = re.sub(r'\s*,\s*\n\s*\]', '', params)
        params = re.sub(r'\s*:\s*\n\s*,', ':', params)
        return f"{}def {}({}):"

    content = re.sub(r'(\s*)def\s+([^\n(]+)\(\s*([^)]+)\)\s*:', fix_method, content)
    return content

def fix_dict_creation(content):
    # Fix dictionary creation syntax
    def fix_dict(match):
        indent = match.group(1)
        content = match.group(2)
        # Clean up dictionary content
        content = re.sub(r':\s*ste,\s*p\s*"', ': step, "', content)
        content = re.sub(r'\*\*metrics,\s*\n', '**metrics\n', content)
        return f"{}{}    {}\n{}}}"

    content = re.sub(r'(\s*)log_entry\s*=\s*{}]+)\s*}', fix_dict, content)
    return content

def fix_file_operations(content):
    # Fix file operation syntax
    content = re.sub(r'open\(([^)]+)\s+"([^"]+)"\)', r'open(\1,, "\2")', content)
    return content

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_imports(content)
        content = fix_docstrings(content)
        content = fix_method_definitions(content)
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
