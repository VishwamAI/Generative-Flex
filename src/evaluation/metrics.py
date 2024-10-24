"""
Core Evaluation Metrics for Generative-Flex
Implements essential metrics for model evaluation and benchmarking
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import Perplexity
import logging


@dataclass
class EvalMetrics:
    """Collection of evaluation metrics"""

    perplexity: float
    bleu: Optional[float] = None
    rouge: Optional[Dict[str, float]] = None


class CoreEvaluator:
    """Core evaluator with essential metrics"""

    def __init__(self, device: torch.device):
        self.device = device
        self.setup_metrics()

    def setup_metrics(self):
        """Setup core evaluation metrics"""
        self.perplexity = Perplexity(ignore_index=-100).to(self.device)
        self.bleu = BLEUScore(n_gram=4).to(self.device)
        self.rouge = ROUGEScore().to(self.device)

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        generated_texts: Optional[List[str]] = None,
        reference_texts: Optional[List[str]] = None,
    ) -> EvalMetrics:
        """Compute core evaluation metrics"""
        metrics = {}

        # Compute perplexity
        metrics["perplexity"] = self.perplexity(
            predictions.view(-1, predictions.size(-1)), labels.view(-1)
        ).item()

        # Compute generation metrics if texts are provided
        if generated_texts and reference_texts:
            metrics["bleu"] = self.bleu(
                generated_texts, [[ref] for ref in reference_texts]
            ).item()

            rouge_scores = self.rouge(generated_texts, reference_texts)
            metrics["rouge"] = {k: v.item() for k, v in rouge_scores.items()}

        return EvalMetrics(**metrics)

    def log_metrics(self, metrics: EvalMetrics, step: int):
        """Log metrics to console"""
        logging.info(f"Step {step} Evaluation Metrics:")
        logging.info(f"Perplexity: {metrics.perplexity:.4f}")

        if metrics.bleu is not None:
            logging.info(f"BLEU: {metrics.bleu:.4f}")

        if metrics.rouge is not None:
            for k, v in metrics.rouge.items():
                logging.info(f"ROUGE-{k}: {v:.4f}")
