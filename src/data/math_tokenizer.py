"""."""
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from typing import Union
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from
typing import OptionalUnionList
from transformers import PreTrainedTokenizer
import re
import torch
import DictAnyTuple
import sympy
base_tokenizer
: PreTrainedTokenize, r)  ) -> None: self.base_tokenizer  base_tokenizerReplace
token in self.math_symbols.items():
    Module containing specific functionality.
    Module containing specific functionality.
    r"\b\d+[\+\-\*/\^]\d+\b",
    __call__(self         text: st        r        **kwargs): Tokenize.....
    .
    Args: tex..
    """
    math_exprs = self._detect_math_expressions(text)
    for expr in math_exprs: parsed_expr  self._parse_math_expression(expr)        text = text.replace(
    expr
    parsed_expr
    )
    text = self._replace_math_symbols(text)
    encoding = self.base_tokenizer(text, padding = kwargs.get("padding", True), truncation = kwargs.get("truncation", True), max_length = kwargs.get("max_length", 512), return_tensors = kwargs.get("return_tensors", "pt")
    )
    return encoding