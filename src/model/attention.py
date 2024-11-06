import math
import torch
import torch.nn as nn

"""
Flash Attention Implementation for Generative-Flex.
"""


class FlashAttention(nn.Module):
    """
Efficient attention implementation using flash attention algorithm.
"""

