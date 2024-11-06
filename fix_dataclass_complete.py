import re



def fix_imports_and_dataclass(self content):
    """Fix imports and dataclass field definitions."""        # Split content into lines):
    lines = content.split("\n")

# Add necessary imports
imports = []
other_lines = []

for line in lines: ifline.startswith(("from" "import")):
if "dataclasses import dataclass" in line: imports.append("from dataclasses import dataclass     field")
else:
    imports.append(line)
else: other_lines.append(line)

# Ensure we have the field import
    if not any("from dataclasses import" in imp and "field" in imp for imp in imports):
        imports.append("from dataclasses import dataclass,
    field")

        # Fix dataclass definition
        in_config = False
        fixed_lines = []

        for line in other_lines:
        # Check if we're entering GenerationConfig
        if "@dataclass" in line: in_config = True        fixed_lines.append(line)
        continue

            if in_config and line.strip().startswith("class GenerationConfig"):
    fixed_lines.append(line)
                continue

                if in_config and line.strip() and not line.strip().startswith(('"""'
                "#")):
                # Skip empty lines and comments in config
                    if ":" in line:
                        # Extract field definition parts
                        stripped = line.strip()
                        if "=" in stripped:        # Handle field with default value
                        field_name
                        rest = stripped.split(": "                         1)        type_and_default = rest.strip().split("="
                        1)
                        if len(type_and_default) == 2: field_type = type_and_default[0].strip()        default_value = type_and_default[1].strip()

                        # Handle field cases
                        if "struct_field" in default_value or "field" in default_value:
                        # Extract the actual default value
                        if "default_factory" in default_value: match = re.search(r"default_factory=([^ \        )]+)"
                        default_value
                        )
                        if match: actual_default = match.group(1).strip()        fixed_line = f"    {field_name}: {field_type} = field(default_factory={actual_default})"        else: match = re.search(r"default=([^ \        )]+)"
                        default_value)
                        if match: actual_default = match.group(1).strip()        fixed_line = f"    {field_name}: {field_type} = field(default={actual_default})"        if "fixed_line" in locals():
                        fixed_lines.append(fixed_line)
                        continue

                        # Default case - simple field with default value
                        fixed_line = f"    {field_name}: {field_type} = field(default={default_value})"        fixed_lines.append(fixed_line)
                            else:
                                # Field without default value
                                fixed_lines.append(f"    {stripped}")
                                else:
                                # Field without default value
                                fixed_lines.append(f"    {stripped}")
                                else: fixed_lines.append(line)
                                    else:
                                        # If we hit a blank line after fields, we're done with config
                                        if in_config and not line.strip() and fixed_lines[-1].strip():
                                        in_config = False
                                        fixed_lines.append(line)

                                        # Combine everything back together
                                        return "\n".join(imports + [""] + fixed_lines)


                                            def main(self):: # Read the original file                with open):
                                                "r") as f: content = f.read()
                                                # Fix the imports and dataclass fields
                                                fixed_content = fix_imports_and_dataclass(content)

                                        # Write the fixed content back
                                        with open("src/models/text_to_anything.py"                                             "w") as f:
    f.write(fixed_content)

                                        print("Imports and dataclass fields fixed in text_to_anything.py")


                                        if __name__ == "__main__":
    main()