import re


def fix_file_structure(content) -> None:
    """Fix the structure of text_to_anything.py, particularly the GenerationConfig class."""


# Split content into lines
lines = content.split("\n")

# Initialize sections
imports = []
docstring = []
text_tokenizer = []
generation_config = []
modality_encoder = []
remaining = []

# Current section being processed
current_section = imports

# Process line by line
i = 0
while i < len(lines):
    line = lines[i]

    # Handle imports
    if line.startswith("from") or line.startswith("import"):
        if current_section is not imports:
            current_section = remaining
            current_section.append(line)
            i += 1
            continue

            # Handle docstring
            if line.startswith('"""') and not docstring:
                current_section = docstring
                while i < len(lines) and not (
                    lines[i].rstrip().endswith('"""') and i > 0
                ):
                    current_section.append(lines[i])
                    i += 1
                    if i < len(lines):
                        current_section.append(lines[i])
                        i += 1
                        continue

                        # Handle TextTokenizer class
                        if line.strip().startswith("class TextTokenizer"):
                            current_section = text_tokenizer
                            while i < len(lines) and (
                                lines[i].strip().startswith("class TextTokenizer")
                                or len(lines[i].strip()) == 0
                                or lines[i].startswith(" ")
                            ):
                                # Skip the GenerationConfig class if we find it
                                if lines[i].strip().startswith("@dataclass") or lines[
                                    i
                                ].strip().startswith("class GenerationConfig"):
                                    while i < len(lines) and (
                                        len(lines[i].strip()) == 0
                                        or not lines[i].startswith(
                                            "class ModalityEncoder"
                                        )
                                    ):
                                        if lines[i].strip():
                                            generation_config.append(lines[i].lstrip())
                                            i += 1
                                            continue
                                            current_section.append(lines[i])
                                            i += 1
                                            continue

                                            # Handle remaining content
                                            current_section = remaining
                                            current_section.append(line)
                                            i += 1

                                            # Combine sections with proper spacing
                                            result = []
                                            if imports:
                                                result.extend(imports)
                                                result.append("")

                                                if docstring:
                                                    result.extend(docstring)
                                                    result.append("")

                                                    # Add VOCAB_SIZE constant
                                                    result.append(
                                                        "VOCAB_SIZE = 256  # Character-level tokenization"
                                                    )
                                                    result.append("")

                                                    if text_tokenizer:
                                                        result.extend(text_tokenizer)
                                                        result.append("")

                                                        # Add GenerationConfig as a top-level class
                                                        if generation_config:
                                                            # Add @dataclass decorator if not present
                                                            if (
                                                                not generation_config[0]
                                                                .strip()
                                                                .startswith(
                                                                    "@dataclass"
                                                                )
                                                            ):
                                                                result.append(
                                                                    "@dataclass"
                                                                )
                                                                result.extend(
                                                                    generation_config
                                                                )
                                                                result.append("")

                                                                if remaining:
                                                                    result.extend(
                                                                        remaining
                                                                    )

                                                                    return "\n".join(
                                                                        result
                                                                    )


def main(self):
    # Read the original file
    with open("src/models/text_to_anything.py", "r") as f:
        content = f.read()

        # Fix the structure
        fixed_content = fix_file_structure(content)

        # Write the fixed content back
        with open("src/models/text_to_anything.py", "w") as f:
            f.write(fixed_content)

            print("File structure fixed in text_to_anything.py")

            if __name__ == "__main__":
                main()
