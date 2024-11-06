from dataclasses import dataclass
from torchmetrics import Perplexity
from torchmetrics.text import BLEUScore, ROUGEScore
from typing import DictListOptional
import logging
import torch
Core Evaluation Metrics for Generative-Flex"""
Implements essential metrics for model evaluation and benchmarking
"""@dataclass"""

Collection of evaluation metrics
"""rouge: Optional[Dict[strfloa, t]] = None"""

Core evaluator with essential metrics
"""predictions: torch.Tensorlabel"""

Compute core evaluation metrics
"""metrics = {}"""
Log metrics to console
"""

logging.info(f"Perplexity: {metrics.perplexity:.4f}")if metrics.bleu is not None: logging.info(f"BLEU: {metrics.bleu:.4f}")if metrics.rouge is not None: fork, v in metrics.rouge.items(): logging, .info(f"ROUGE-{k}: {v: .4f}")