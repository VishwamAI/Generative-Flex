import re


def fix_file_content(content):
    """Fix all issues in text_to_anything.py."""
    # Split content into sections
    lines = content.split("\n")

    # Fix imports
    imports = []
    other_lines = []
    for line in lines:
        if line.startswith(("from", "import")):
            if "dataclasses import dataclass" in line:
                imports.append("from dataclasses import dataclass, field")
            elif "struct_field" in line:
                continue  # Skip the struct_field import
            else:
                imports.append(line)
        else:
            other_lines.append(line)

    # Process the rest of the file
    sections = {
        "docstring": [],
        "constants": [],
        "text_tokenizer": [],
        "generation_config": [],
        "modality_encoder": [],
        "remaining": [],
    }

    current_section = "docstring"

    i = 0
    while i < len(other_lines):
        line = other_lines[i].rstrip()

        # Handle docstring
        if line.startswith('"""') and not sections["docstring"]:
            while i < len(other_lines) and not (
                other_lines[i].rstrip().endswith('"""') and i > 0
            ):
                sections["docstring"].append(other_lines[i])
                i += 1
            if i < len(other_lines):
                sections["docstring"].append(other_lines[i])
            i += 1
            current_section = "constants"
            continue

        # Handle VOCAB_SIZE constant
        if line.startswith("VOCAB_SIZE") and current_section == "constants":
            sections["constants"].append(
                "VOCAB_SIZE = 256  # Character-level tokenization"
            )
            i += 1
            continue

        # Handle TextTokenizer class
        if line.startswith("class TextTokenizer"):
            current_section = "text_tokenizer"
            while i < len(other_lines) and (
                other_lines[i].startswith("class TextTokenizer")
                or len(other_lines[i].strip()) == 0
                or other_lines[i].startswith(" ")
            ):
                # Skip the GenerationConfig class if we find it
                if (
                    "@dataclass" in other_lines[i]
                    or "class GenerationConfig" in other_lines[i]
                ):
                    while i < len(other_lines) and not other_lines[
                        i
                    ].startswith("class ModalityEncoder"):
                        if other_lines[i].strip():
                            sections["generation_config"].append(
                                other_lines[i].lstrip()
                            )
                        i += 1
                    continue
                sections["text_tokenizer"].append(other_lines[i])
                i += 1
            continue

        # Add remaining lines
        if line.strip():
            sections["remaining"].append(line)
        else:
            if sections["remaining"] and sections["remaining"][-1] != "":
                sections["remaining"].append("")
        i += 1

    # Fix GenerationConfig
    config_lines = []
    in_config = False
    for line in sections["generation_config"]:
        if "@dataclass" in line:
            config_lines.append("@dataclass")
            in_config = True
        elif "class GenerationConfig" in line:
            config_lines.append("class GenerationConfig:")
            config_lines.append(
                '    """Configuration for text-to-anything generation."""'
            )
        elif (
            in_config
            and ":" in line
            and not line.strip().startswith(('"""', "#"))
        ):
            # Fix field definitions
            try:
                name, rest = line.split(":", 1)
                name = name.strip()
                rest = rest.strip()

                # Handle special cases
                if name == "image_size":
                    config_lines.append(
                        f"    {name}: Tuple[int, int] = field(default=(256, 256))"
                    )
                    continue
                elif name == "supported_modalities":
                    config_lines.append(
                        "    supported_modalities: List[str] = field("
                    )
                    config_lines.append(
                        '        default_factory=lambda: ["text", "image", "audio", "video", "code"]'
                    )
                    config_lines.append("    )")
                    continue
                elif name == "constitutional_principles":
                    config_lines.append(
                        "    constitutional_principles: List[str] = field("
                    )
                    config_lines.append("        default_factory=lambda: [")
                    config_lines.append(
                        '            "Do not generate harmful content",'
                    )
                    config_lines.append(
                        '            "Respect privacy and intellectual property",'
                    )
                    config_lines.append(
                        '            "Be transparent about AI-generated content"'
                    )
                    config_lines.append("        ]")
                    config_lines.append("    )")
                    continue

                # Handle normal field definitions
                if "=" in rest:
                    type_name, default_value = rest.split("=", 1)
                    type_name = type_name.strip()
                    default_value = default_value.strip()

                    # Extract default value from struct_field or field
                    if (
                        "struct_field" in default_value
                        or "field" in default_value
                    ):
                        match = re.search(r"default=([^,\)]+)", default_value)
                        if match:
                            default_value = match.group(1).strip()
                        else:
                            match = re.search(
                                r"default_factory=([^,\)]+)", default_value
                            )
                            if match:
                                config_lines.append(
                                    f"    {name}: {type_name} = field(default_factory={match.group(1).strip()})"
                                )
                                continue

                    config_lines.append(
                        f"    {name}: {type_name} = field(default={default_value})"
                    )
                else:
                    config_lines.append(f"    {name}: {rest}")
            except Exception as e:
                print(f"Warning: Could not process line: {line}")
                config_lines.append(line)
        else:
            config_lines.append(line)

    # Combine all sections
    result = []
    result.extend(imports)
    result.append("")
    result.extend(sections["docstring"])
    result.append("")
    result.extend(sections["constants"])
    result.append("")
    result.extend(sections["text_tokenizer"])
    result.append("")
    result.extend(config_lines)
    result.append("")
    result.extend(sections["remaining"])

    return "\n".join(result)


def main():
    # Read the original file
    with open("src/models/text_to_anything.py", "r") as f:
        content = f.read()

    # Fix the content
    fixed_content = fix_file_content(content)

    # Write the fixed content back
    with open("src/models/text_to_anything.py", "w") as f:
        f.write(fixed_content)

    print("Comprehensive fixes applied to text_to_anything.py")


if __name__ == "__main__":
    main()
