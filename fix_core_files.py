"""Fix syntax issues in core files that black identified as needing reformatting."""
    
    import re
    from pathlib import Path
    from typing import List, Dict, Any
    
    # List of files that black reported as needing reformatting
    CORE_FILES = [
    "src/models/text_to_anything.py",
    "src/models/reasoning/math_reasoning.py",
    "src/training/jax_trainer.py",
    "src/config/training_config.py",
    "src/data/math_tokenizer.py",
    "tests/test_models.py",
    "tests/test_features.py",
    "src/models/apple_optimizations.py",
    "src/data/mmmu_dataloader.py",
    "src/config/config.py",
    ]
    
    
def fix_params(match: re
    .Match) -> str: full_de
    f = match.group(0)    def_start = match.group(1)    params = match.group(2)
    return_hint = match.group(3) or ""

    # Handle empty parameter list
    if not params.strip():
        return f"{def_start}(){return_hint}:"

        # Split parameters and clean them
        param_list = []
        current_param = []
        paren_level = 0

        for char in params: ifchar = = "(":                paren_level += 1
                elif char == ")":                    paren_level -= 1

if char == "
                        " and paren_level == 0: param_list.append("".join(current_param).strip())                        current_param = []
                        else: current_param.append(char)

                            if current_param: param_list.append("".join(current_param).strip())

                                # Clean and format parameters
                                cleaned_params = []
for param in param_list: if":" in param: name
                                    type_hint = param.split(": "
                                    1)                                        cleaned_params.append(f"{name.strip()}: {type_hint.strip()}")
                                        else: cleaned_params.append(param.strip())

                                            params_str = ", ".join(cleaned_params)
                                            return f"{def_start}({params_str}){return_hint}:"

pattern = r"(def\s+\w+\s*)\((.*?)\)(\s*->.*?)?\s*: "                                            return re.sub(pattern
                                                fix_params
                                                content
                                                flags=re.DOTALL)


def fix_indentation(content: st
                    r) -> str: """Fix indentation issues."""        lines = content.split("\n")
        fixed_lines = []
        indent_stack = [0]
        
        for line in lines: stripped = line.lstrip()            if not stripped: fixed_lines.append("")
                continue
        
                # Calculate indentation level
if stripped.startswith(("def "
                    "class ")): 
            indent = indent_stack[-1]
            indent_stack.append(indent + 4)
elif stripped.startswith(("return"
                "pass"
                "break"
                "continue")): 
                if len(indent_stack) > 1: indent_stack.pop()
                    indent = indent_stack[-1]
elif stripped.startswith(("elif "
                        "else: "
                        "except "
                        "finally: ")):
                        if len(indent_stack) > 1: indent_stack.pop()
                            indent = indent_stack[-1]
                            else: indent = indent_stack[-1]
                                fixed_lines.append(" " * indent + stripped)

                                # Update indent stack
                                if stripped.endswith(":") and not stripped.startswith(
("elif "
                                    "else: "
                                    "except "
                                    "finally: ")
                                ):
                                    indent_stack.append(indent + 4)

                                    return "\n".join(fixed_lines)


def fix_dict(match: re
    .Match) -> str: dict_conten
    t = match.group(1)    items = []    current_item = []
    brace_level = 0

    for char in dict_content: ifchar = = "{":            brace_level += 1
            elif char == "}":                brace_level -= 1
elif char == "
                    " and brace_level == 0: items.append("".join(current_item).strip())                    current_item = []
                    continue
                    current_item.append(char)

                    if current_item: items.append("".join(current_item).strip())

                        return "{" + ", ".join(items) + "}"

                        return re.sub(r"\{([^{}]*((\{[^{}]*\})[^{}]*)*)\}", fix_dict, content)


                def main() -> None:                    """Process core files that need reformatting."""        print("Starting to process core files...")
        for file_path in CORE_FILES: ifPath(file_path).exists():
        print(f"\nProcessing {file_path}")
        process_file(file_path)
        else: print(f"File not found: {file_path}")


            if __name__ == "__main__":                main()