import os



def fix_file_syntax(filename) -> None:
    with open(filename, "r") as f:
        lines = f.readlines()

        # Track if we made any changes
        modified = False
        new_lines = []
        i = 0

        while i < len(lines):
        line = lines[i]

        # Fix line continuation issues
        if line.strip().endswith(", ") or line.strip().endswith("("):
            # Look ahead to see if next line is improperly indented
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                current_indent = len(line) - len(line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                # If next line isn't properly indented, fix it
                if next_indent <= current_indent:
                    modified = True
                    new_lines.append(line.rstrip() + "\n")
                    new_lines.append(" " * (current_indent + 4) + next_line.lstrip())
                    i += 2
                    continue

                # Fix specific issues found in the error messages
                if "config.max_position_embeddings" in line:
                    modified = True
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + "config.max_position_embeddings, \n")
                    elif "self.config.max_sequence_length" in line:
                        modified = True
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(" " * indent + "self.config.max_sequence_length, \n")
                        elif "config.hidden_size, 256" in line:
                            modified = True
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(" " * indent + "config.hidden_size, \n")
                            new_lines.append(" " * indent + "256, \n")
                            elif "generation_config.num_attention_heads * 8" in line:
                                modified = True
                                indent = len(line) - len(line.lstrip())
                                new_lines.append(" " * indent + "generation_config.num_attention_heads * 8, \n")
                                else:
                                    new_lines.append(line)
                                    i += 1

                                    if modified:
                                        print(f"Fixing syntax in {filename}")
                                        with open(filename, "w") as f:
                                            f.writelines(new_lines)


                                            def main():
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
