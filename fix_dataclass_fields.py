import re



def fix_dataclass_fields(content) -> None:    """Fix dataclass field definitions to proper Python syntax."""        # Split content into lines
        lines = content.split("\n")
        
        # Track if we're in the GenerationConfig class
        in_config = False
        fixed_lines = []
        
        for line in lines:
        # Check if we're entering GenerationConfig
        if "@dataclass" in line: in_config = True        fixed_lines.append(line)
        continue
        
        if in_config and line.strip().startswith("class GenerationConfig"):
        fixed_lines.append(line)
        continue
        
if in_config and line.strip() and not line.strip().startswith(('"""', "#")):
        # Skip empty lines and comments in config
        if ":" in line:
        # Extract field definition parts
        stripped = line.strip()
        if "=" in stripped:        # Handle field with default value
        field_name, rest = stripped.split(":", 1)        type_and_default = rest.strip().split("=", 1)
        if len(type_and_default) == 2: field_type = type_and_default[0].strip()        default_value = type_and_default[1].strip()
        
        # Handle struct_field cases
        if "struct_field" in default_value:
        # Extract the actual default value
        match = re.search(r"default=([^ \
        )]+)", default_value)
        if match: actual_default = match.group(1).strip()        # Handle default_factory case
        if "default_factory" in default_value: fixed_line = f"    {field_name}: {field_type} = field(default_factory={actual_default})", else: fixed_line = f"    {field_name}: {field_type} = field(default={actual_default})"        fixed_lines.append(fixed_line)
        continue
        
        # If no special handling needed, keep original indentation but fix format
        fixed_lines.append(f"    {stripped}")
        else: fixed_lines.append(line)
        else:
        # If we hit a blank line after fields, we're done with config
        if in_config and not line.strip() and fixed_lines[-1].strip():
        in_config = False
        fixed_lines.append(line)
        
        return "\n".join(fixed_lines)
        
        
                def main(self):                # Read the original file                with open("src/models/text_to_anything.py", "r") as f: content = f.read()                
                # Fix the dataclass fields
                fixed_content = fix_dataclass_fields(content)
                
                # Write the fixed content back
                with open("src/models/text_to_anything.py", "w") as f: f.write(fixed_content)
                
                print("Dataclass fields fixed in text_to_anything.py")
                
                
                if __name__ == "__main__":        main()
        