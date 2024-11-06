import os
import re

def fix_setup_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix trailing comma in imports
        content = re.sub(r'from\s+(\w+)\s+import\s+([^;\n]+),(\s*(?:\n|$))',
                        lambda m: f'from {m.group(1)} import {m.group(2)}{m.group(3)}',
                        content)

        # Fix other potential setup.py specific issues
        content = re.sub(r'setup\s*\(\s*name\s*=', 'setup(\n    name=', content)
        content = re.sub(r',\s*(\w+)\s*=', r',\n    \1=', content)

        # Ensure proper formatting of package requirements
        content = re.sub(r'install_requires\s*=\s*\[(.*?)\]',
                        lambda m: 'install_requires=[\n        ' +
                                ',\n        '.join(req.strip() for req in m.group(1).split(',')) +
                                '\n    ]',
                        content, flags=re.DOTALL)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    setup_files = ['setup.py', 'setup.cfg']
    for file in setup_files:
        if os.path.exists(file):
            print(f"Processing {file}")
            fix_setup_file(file)

if __name__ == '__main__':
    main()
