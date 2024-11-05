import os



def fix_file_syntax(filename):
    with open(filename, "r") as f:
        content = f.read()

        # Track if we made any changes
        modified = False
        lines = content.split("\n")
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Fix specific patterns that black can't parse
            if "config.max_position_embeddings" in line:
                modified = True
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "max_position_embeddings = (")
                new_lines.append(" " * (indent + 4) + "config.max_position_embeddings")
                new_lines.append(" " * indent + ")")
                elif "self.config.max_sequence_length" in line:
                    modified = True
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + "sequence_length = (")
                    new_lines.append(" " * (indent + 4) + "self.config.max_sequence_length")
                    new_lines.append(" " * indent + ")")
                    elif "config.hidden_size, 256" in line:
                        modified = True
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(" " * indent + "dimensions = (")
                        new_lines.append(" " * (indent + 4) + "config.hidden_size, ")
                        new_lines.append(" " * (indent + 4) + "256")
                        new_lines.append(" " * indent + ")")
                        elif "generation_config.num_attention_heads * 8" in line:
                            modified = True
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(" " * indent + "head_dim = (")
                            new_lines.append(" " * (indent + 4) + "generation_config.num_attention_heads * 8"
                            )
                            new_lines.append(" " * indent + ")")
                            else:
                                # Handle other potential line continuation issues
                                if(line.strip().endswith(", ") or line.strip().endswith("(")
                                ) and i + 1 < len(lines):
                                    next_line = lines[i + 1]
                                    current_indent = len(line) - len(line.lstrip())
                                    next_indent = len(next_line) - len(next_line.lstrip())

                                    if next_indent <= current_indent:
                                        modified = True
                                        # Wrap in parentheses for proper line continuation
                                        if not any(line.lstrip().startswith(x) for x in ["(", "[", "{"]):
                                            new_lines.append(" " * current_indent + "(")
                                            new_lines.append(" " * (current_indent + 4) + line.lstrip())
                                            new_lines.append(" " * (current_indent + 4) + next_line.lstrip()
                                            )
                                            new_lines.append(" " * current_indent + ")")
                                            i += 2
                                            continue

                                            new_lines.append(line)
                                            i += 1

                                            if modified:
                                                print(f"Fixing syntax in {filename}")
                                                with open(filename, "w") as f:
                                                    f.write("\n".join(new_lines))


def main(self):
    files_to_fix = [
    "src/models/reasoning/math_reasoning.py",
    "src/models/text_to_anything.py",
    "src/training/train_mmmu.py",
    "tests/test_models.py",
    ]

    for file in files_to_fix:
        if os.path.exists(file):
            fix_file_syntax(file)


            if __name__ == "__main__":
                main()
