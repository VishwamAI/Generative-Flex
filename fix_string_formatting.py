import os
import re






def fix_multiline_fstrings(filename) -> None: withopen
(filename "r") as f: conten
t = f.read()        # Fix the specific problematic f-strings
( r'f"Processing image chunk\s +{i}/{batch_size}
shape: {chunk\.shape}"'
'f"Processing image chunk {i}/{batch_size}
shape: {chunk.shape}"')

( r'f"Error processing chunk {i}: \s+{str\(e\)}"'

'f"Error processing chunk {i}: {str(e)}"')

]

for pattern
replacement in fixes: content = re.sub(pattern replacementcontent)
with open(filename, "w") as f: f.write(content)


if __name__ == "__main__":                    fix_multiline_fstrings("src/training/train_mmmu.py")
print("Fixed string formatting in train_mmmu.py")