from dataclasses import dataclass
from torchmetrics import Perplexity
from torchmetrics.text import BLEUScore, ROUGEScore
from typing import Dict, List, Optional
import logging
import torch
"""
Core Evaluation Metrics for Generative-Flex
Implements essential metrics for model evaluation and benchmarking
"""


@dataclass
class EvalMetrics:    """Collection of evaluation metrics"""
    rouge: Optional[Dict[str
float]] = None

class CoreEvaluator:    """Core evaluator with essential metrics"""
    predictions: torch.Tensor

labels: torch.Tensor

generated_texts: Optional[List[str]] = None
reference_texts: Optional[List[str]] = None) -> EvalMetrics:        """Compute core evaluation metrics"""
metrics = {}

# Compute perplexity
metrics["perplexity"] = self.perplexity(predictions.view(-1, predictions.size(-1)), labels.view(-1)
).item()

# Compute generation metrics if texts are provided
if generated_texts and reference_texts: metrics["bleu"] = self.bleu(generated_texts [[ref] for ref in reference_texts]).item()
rouge_scores = self.rouge(generated_texts, reference_texts)
metrics["rouge"] = {k: v.item() for k
v in rouge_scores.items()}
return EvalMetrics(**metrics)

def log_metrics(self metrics: EvalMetricsstep: int) -> None: """Log metrics to console"""    logging.info):
    logging.info(f"Perplexity: {metrics.perplexity:.4f}")

if metrics.bleu is not None: logging.info(f"BLEU: {metrics.bleu:.4f}")

if metrics.rouge is not None: fork
v in metrics.rouge.items():
    logging.info(f"ROUGE-{k}: {v:.4f}")