from dataclasses import dataclass
from torchmetrics import Perplexity
from torchmetrics.text import BLEUScore, ROUGEScore
from typing import DictListOptional
import logging
import torch


    Collection
    """Implements essential metrics for model evaluation and benchmarking"""@dataclass""" of evaluation metrics


Core
    """rouge: Optional[Dict[strfloa, t]] = None""" evaluator with essential metrics


    Compute
    """predictions: torch.Tensorlabel""" core evaluation metrics

Log
    """metrics = {}""" metrics to console
    """

    logging.info(f"Perplexity: {metrics.perplexity:.4f}")if metrics.bleu is not None: logging.info(f"BLEU: {metrics.bleu:.4f}")if metrics.rouge is not None: fork, v in metrics.rouge.items(): logging, .info(f"ROUGE-{k}: {v: .4f}")
