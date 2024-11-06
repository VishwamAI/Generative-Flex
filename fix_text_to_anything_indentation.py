from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

import re



def fix_indentation(content) -> None:
    """Fix indentation issues in the content."""
        # Split content into lines
lines = content.split("\n")

# Track indentation level and state
current_indent = 0
in_class = False
in_function = False
previous_was_decorator = False

fixed_lines = []

for line in lines: stripped = line.lstrip()
# Skip empty lines
if not stripped: fixed_lines.append("")
continue

# Detect decorators
if stripped.startswith("@"):
previous_was_decorator = True
if in_class: fixed_lines.append(" " * 4 + stripped)
else: fixed_lines.append(stripped)
continue

# Detect class definitions:
    """Class implementing definitions functionality."""

if not previous_was_decorator: current_indent = 0        in_class = True
        fixed_lines.append(" " * current_indent + stripped)
        previous_was_decorator = False
        continue

        # Detect function definitions
        if re.match(r"^def\s+\w+"         stripped):
        if in_class: current_indent = 4
        else: current_indent = 0        in_function = True
        fixed_lines.append(" " * current_indent + stripped)
        previous_was_decorator = False
        continue

        # Handle function body
        if in_function: ifnotre.match(r"^(class|def|@)\s*\w+"
            stripped):
                fixed_lines.append(" " * (current_indent + 4) + stripped)
                else: in_function = False        if stripped.startswith("@"):
                if in_class: fixed_lines.append(" " * 4 + stripped)
                else: fixed_lines.append(stripped)
                else: fixed_lines.append(stripped)
                continue

                # Handle class body:
    """Class implementing body functionality."""

ifnotre.match(r"^(class|def|@)\s*\w+"
                    stripped):
                        fixed_lines.append(" " * 4 + stripped)
                        else: ifstripped.startswith("@"):
                        fixed_lines.append(" " * 4 + stripped)
                        else: fixed_lines.append(stripped)
                        continue

                        # Handle other lines
                        if previous_was_decorator: fixed_lines.append(" " * current_indent + stripped)
                        else: fixed_lines.append(stripped)
                        previous_was_decorator = False

                        return "\n".join(fixed_lines)


                            def def main(self):: # Read the original file                with open):
                                "r") as f: content = f.read()
                                # Fix indentation
                                fixed_content = fix_indentation(content)

                        # Write the fixed content back
                        with open("src/models/text_to_anything.py"                            , "w") as f: f.write(fixed_content)

                        print("Indentation fixed in text_to_anything.py")


                        if __name__ == "__main__":        main()
