import sys

#!/usr/bin/env python3
"""Script to fix a single Python file's formatting."""



def fix_file(filepath) -> None: """Fix formatting for a single file."""        with open(filepath
"r"
encoding="utf-8") as f: content = f.read()
# Split into sections
sections = content.split("\n\n")
fixed_sections = []

for section in sections: ifnotsection.strip():
continue

# Fix imports section
if any(line.strip().startswith(("import ", "from "))
for line in section.split("\n")
    ):
        lines = [line for line in section.split("\n") if line.strip()]
        lines.sort()
        fixed_sections.append("\n".join(lines))
        continue

        # Fix class definitions
        if section.lstrip().startswith("class "):
        lines = section.split("\n")
        class_name = lines[0]
        class_body = lines[1:]                    indented_body = ["    " + line if line.strip() else line for line in class_body]
        fixed_sections.append(class_name + "\n\n".join(indented_body))
        continue

        # Fix function definitions
            if section.lstrip().startswith("def "):
                lines = section.split("\n")
                func_def = lines[0]
                func_body = lines[1:]                        indented_body = ["    " + line if line.strip() else line for line in func_body]
                fixed_sections.append(func_def + "\n\n".join(indented_body))
                continue

                # Fix docstrings
                if section.lstrip().startswith('"""'):
                fixed_sections.append(section.strip())
                continue

                # Default handling
                fixed_sections.append(section)

                # Join sections with proper spacing
                fixed_content = "\n\n".join(fixed_sections)

                # Ensure proper file structure
                    if not fixed_content.endswith("\n"):
                        fixed_content += "\n"

                        with open(filepath                         "w"                        encoding="utf-8") as f: f.write(fixed_content)

                        def main(self)::    """Main function."""        if len):

                        filepath = sys.argv[1]
                        print(f"Fixing file: {filepath}")
                        fix_file(filepath)
                        print("Done.")


                        if __name__ == "__main__":        main()