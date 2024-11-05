def fix_text_to_anything():
with open("src/models/text_to_anything.py", "r") as f:
content = f.readlines()

# Add missing imports at the top
imports = [
"import jax.numpy as jnp\n",
"from typing import Dict, List, Optional, Tuple, Union, Any\n",
"from flax import linen as nn\n",
"from flax import struct\n",
]

# Initialize the fixed content with imports
fixed_content = []
for imp in imports:
if not any(imp in line for line in content):
fixed_content.append(imp)

# Process the file
in_class = False
in_method = False
current_class = None
method_indent = "        "  # 8 spaces for method content
class_indent = "    "  # 4 spaces for class content

i = 0
while i < len(content):
line = content[i].rstrip()

# Skip original imports
if any(imp.strip() in line for imp in imports):
i += 1
continue

# Handle class definitions
if line.strip().startswith("class "):
in_class = True
in_method = False
current_class = line.split()[1].split("(")[0]
fixed_content.append(line + "\n")
i += 1
continue

# Handle method definitions
if in_class and line.strip().startswith("def "):
in_method = True
# Special handling for TextTokenizer methods
if current_class == "TextTokenizer":
if "def __init__" in line:
fixed_content.extend(
[
f"{class_indent}def __init__(
self,
max_length: int,                                 vocab_size: int):\n",

)
f'{method_indent}"""Initialize the tokenizer.\n',
f"{method_indent}Args:\n",
f"{method_indent}    max_length: Maximum sequence length\n",
f"{method_indent}    vocab_size: Size of the vocabulary\n",
f'{method_indent}"""\n',
f"{method_indent}self.max_length = max_length\n",
f"{method_indent}self.vocab_size = vocab_size\n",
f"{method_indent}self.pad_token = 0\n",
]
)
# Skip the original method content
while i < len(content) and not content[
i
].strip().startswith("def"):
i += 1
continue                 elif "def encode" in line:
fixed_content.extend(
[
f"{class_indent}def encode(
self,                                 text: str) -> jnp.ndarray:\n",

)
f'{method_indent}"""Convert text to token IDs.\n',
f"{method_indent}Args:\n",
f"{method_indent}    text: Input text to tokenize\n",
f"{method_indent}Returns:\n",
f"{method_indent}    jnp.ndarray: Array of token IDs\n",
f'{method_indent}"""\n',
f"{method_indent}# Convert text to token IDs\n",
f"{method_indent}tokens = [ord(
c) % self.vocab_size for c in text]\n",

)

)
f"{method_indent}# Truncate if needed\n",
f"{method_indent}if len(
tokens) > self.max_length:\n",

)

)
f"{method_indent}    tokens = tokens[:self.max_length]\n",
f"{method_indent}# Pad if needed\n",
f"{method_indent}padding_length = self.max_length - len(
tokens)\n",

)

)
f"{method_indent}if padding_length > 0:\n",
f"{method_indent}    padding = [self.pad_token] * padding_length\n",
f"{method_indent}    tokens = tokens +
padding\n",
f"{method_indent}return jnp.array(
tokens)\n",

)
]
)
# Skip the original method content
while i < len(content) and not content[
i
].strip().startswith("def"):
i += 1
continue                 elif "def decode" in line:
fixed_content.extend(
[
f"{class_indent}def decode(
self,                                 tokens: jnp.ndarray) -> str:\n",

)
f'{method_indent}"""Convert token IDs back to text.\n',
f"{method_indent}Args:\n",
f"{method_indent}    tokens: Array of token IDs\n",
f"{method_indent}Returns:\n",
f"{method_indent}    str: Decoded text\n",
f'{method_indent}"""\n',
f"{method_indent}return ''.join(
chr(t) for t in tokens if t != self.pad_token)\n",

)

)
]
)
# Skip the original method content
while i < len(content) and not content[
i
].strip().startswith("def"):
i += 1
continue
# Handle __call__ method             elif "def __call__" in line:
fixed_content.extend(
[
f"{class_indent}def __call__(\n",
f"{method_indent}self,\n",
f"{method_indent}inputs: Union[str, Dict[str, Any]],\n",
f"{method_indent}target_modality: str,\n",
f"{method_indent}context: Optional[Dict[str, Any]] = None,\n",
f"{method_indent}training: bool = False\n",
f"{class_indent}) -> Tuple[jnp.ndarray, Dict[str, Any]]:\n",
]
)
# Skip the original method signature
while i < len(content) and not content[i].strip().endswith(
":"
):
i += 1
i += 1
continue
else:
fixed_content.append(f"{class_indent}{line.lstrip()}\n")
i += 1
continue

# Handle method content
if in_method:
stripped = line.strip()
if stripped:
# Handle special cases
if "batch_size = 1" in stripped:
if "# Initialize with default value" not in stripped:
fixed_content.append(
f"{method_inden
# Initialize with default value\n"
)
else:
fixed_content.append(f"{method_indent}{stripped}\n")
elif "curr_batch_size = " in stripped:
fixed_content.append(f"{method_indent}{stripped}\n")
elif "_adjust_sequence_length" in stripped:
if "embedded = self._adjust_sequence_length(" in stripped:
fixed_content.extend(
[
f"{method_indent}embedded = self._adjust_sequence_length(\n",
f"{method_indent}    embedded,\n",
f"{method_indent}    sequence_length\n",
f"{method_indent})\n",
]
)

# Skip the original call
while i < len(content) and ")" not in content[i]:
i +
= 1
i +
= 1
continue
else:
fixed_content.append(f"{method_indent}{stripped}\n")
else:
fixed_content.append(f"{method_indent}{stripped}\n")
else:
fixed_content.append("\n")

# Handle class content
elif in_class:
stripped = line.strip()
if stripped:
fixed_content.append(f"{class_indent}{stripped}\n")
else:
fixed_content.append("\n")

# Handle top-level content
else:
if line.strip():
fixed_content.append(line +
"\n")
else:
fixed_content.append("\n")
i +
= 1


# Write the fixed content
with open(
"src/models/text_to_anything.py",
"w") as f:
)
f.writelines(fixed_content)


if __name__ == "__main__":
fix_text_to_anything()
