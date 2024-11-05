"""Specialized tokenizer for mathematical expressions and symbols."""

from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Optional
import re
import sympy
import torch


class MathTokenizer:
    """Tokenizer for mathematical expressions and symbols."""

    def __init__(self, base_tokenizer: PreTrainedTokenizer):
        """Initialize the math tokenizer.

        Args:
            base_tokenizer: Base HuggingFace tokenizer to extend"""
        self.base_tokenizer = base_tokenizer
        # Calculate available space for special tokens
        vocab_size = len(base_tokenizer)
        max_vocab = 50265  # OPT-125m vocabulary size
        available_tokens = max_vocab - vocab_size

        self.math_symbols = {
            # Basic arithmetic (essential symbols)
            "+": "<ADD>",
            "-": "<SUB>",
            "*": "<MUL>",
            "/": "<DIV>",
            "=": "<EQ>",
            # Greek letters commonly used in math
            "α": "<ALPHA>",
            "β": "<BETA>",
            "γ": "<GAMMA>",
            "π": "<PI>",
            "Σ": "<SIGMA>",
            # Special mathematical symbols
            "∫": "<INTEGRAL>",
            "∂": "<PARTIAL>",
            "∞": "<INFINITY>",
            "√": "<SQRT>",
            "∑": "<SUM>",
            "∏": "<PRODUCT>",
            # Superscripts and subscripts
            "²": "<SUP2>",
            "³": "<SUP3>",
            "₁": "<SUB1>",
            "₂": "<SUB2>",
        }

        # Ensure we don't exceed vocabulary size
        special_tokens = list(self.math_symbols.values())
        if len(special_tokens) > available_tokens:
            # Prioritize basic arithmetic symbols if we need to reduce tokens
            special_tokens = special_tokens[:available_tokens]
            # Update math_symbols to only include tokens we can add
            self.math_symbols = {
                k: v for k, v in self.math_symbols.items() if v in special_tokens
            }

        # Add special tokens to base tokenizer
        self.base_tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

    def _parse_math_expression(self, text: str) -> str:
        """Parse mathematical expressions using sympy."""
        try:
            # Try to parse with sympy
            expr = sympy.parse_expr(text, evaluate=False)
            # Convert to LaTeX for standardized representation
            latex = sympy.latex(expr)
            return latex
        except Exception:
            return text

    def _replace_math_symbols(self, text: str) -> str:
        """Replace mathematical symbols with special tokens."""
        for symbol, token in self.math_symbols.items():
            text = text.replace(symbol, f" {token} ")
        return text

    def _detect_math_expressions(self, text: str) -> List[str]:
        """Detect mathematical expressions in text."""
        # Match expressions between $ signs (LaTeX style)
        math_exprs = re.findall(r"\$(.*?)\$", text)
        # Match expressions with common math patterns
        patterns = [
            r"\b\d+[\+\-\*/\^]\d+\b",  # Basic arithmetic
            r"\b[a-zA-Z]\s*=\s*[-+]?\d*\.?\d+\b",  # Variable assignments
            r"\b\d*\.?\d+\s*[×⋅]\s*\d*\.?\d+\b",  # Multiplication
            r"\b\d*\.?\d+\s*÷\s*\d*\.?\d+\b",  # Division
            r"\b√\d+\b",  # Square roots
            r"\b\d+²\b",  # Squares
            r"\bπ\b",  # Pi
            r"\b∫.*dx\b",  # Integrals
            r"\b∑.*\b",  # Summations
        ]
        for pattern in patterns:
            math_exprs.extend(re.findall(pattern, text))
        return math_exprs

    def tokenize(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text with special handling for mathematical content."""
        # Detect and parse mathematical expressions
        math_exprs = self._detect_math_expressions(text)
        for expr in math_exprs:
            parsed_expr = self._parse_math_expression(expr)
            text = text.replace(expr, parsed_expr)

        # Replace mathematical symbols with special tokens
        text = self._replace_math_symbols(text)

        # Tokenize with base tokenizer
        encoding = self.base_tokenizer(
            text,
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", 512),
            return_tensors=kwargs.get("return_tensors", "pt"),
        )

        return encoding
