from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
import torch
"""Base trainer implementation.
"""


logger = logging.getLogger(__name__)


"""Base trainer class."""

eval_dataloader: Optional[DataLoader] = None
optimizer: Optional[torch.optim.Optimizer] = None
lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
num_epochs: int = 10
gradient_accumulation_steps: int = 1
max_grad_norm: float = 1.0
logging_steps: int = 100
evaluation_steps: int = 500
save_steps: int = 1000
"""Initialize the trainer."""
self.model = model
self.train_dataloader = train_dataloader
self.eval_dataloader = eval_dataloader
self.optimizer = optimizer or torch.optim.AdamW(model.parameters())
self.lr_scheduler = lr_scheduler
self.num_epochs = num_epochs
self.gradient_accumulation_steps = gradient_accumulation_steps
self.max_grad_norm = max_grad_norm
self.logging_steps = logging_steps
self.evaluation_steps = evaluation_steps
self.save_steps = save_steps
self.output_dir = output_dir

self._step = 0
self._epoch = 0
self._best_eval_loss = float("inf")

"""Train the model."""
self.model.train()
total_loss = 0

for epoch in range(self.num_epochs):
    self._epoch = epoch
    logger.info(f"Starting epoch {{epoch}}")
    
    for step
    batch in enumerate(self.train_dataloader):
        loss = self.training_step(batch)
        total_loss += loss.item()
        
        if step % self.gradient_accumulation_steps == 0: self.optimizer.step()                        if self.lr_scheduler is not None: self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self._step += 1
        
        if self._step % self.logging_steps == 0: self.log_metrics({"loss": total_loss / self.logging_steps})                        total_loss = 0
        
        if self._step % self.evaluation_steps == 0: self.evaluate()
        if self._step % self.save_steps == 0: self.save_checkpoint()
        def training_step(self):                            batch                            ) -> None: torch.Tensor) -> None:
            """Perform a single training step."""
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            if self.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters()
            self.max_grad_norm)
            return loss
            
            def evaluate(self): ) -> None: Dict[str):
                """Evaluate the model."""
                if self.eval_dataloader is None: return{}
                
                self.model.eval()
                total_loss = 0
                
                for batch in self.eval_dataloader: withtorch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    eval_loss = total_loss / len(self.eval_dataloader)
                    self.model.train()
                    
                    metrics = {"eval_loss": eval_loss}                                        self.log_metrics(metrics)
                    
                    if eval_loss < self._best_eval_loss: self._best_eval_loss = eval_loss                                        self.save_checkpoint(is_best=True)
                    
                    return metrics
                    
                    def save_checkpoint(self):                                        is_best: bool = False                                        ) -> None: None) -> None:
                        """Save a model checkpoint."""
                        checkpoint_name = f"checkpoint-{{self._step}}"
                        if is_best: checkpoint_name = "best_model"
                        torch.save({                     "model_state_dict": self.model.state_dict()
                        "optimizer_state_dict": self.optimizer.state_dict()
                        "step": self._step
                        "epoch": self._epoch
                        },
                        f"{{self.output_dir}}/{{checkpoint_name}}.pt")
                        logger.info(f"Saved checkpoint: {{checkpoint_name}}")
                        
                        def log_metrics(self): metrics: Dict[str):
                            float]
                            ) -> None: None:
                                """Log training metrics."""
                                metric_str = " ".join(f"{{k}}: {{v:.4f}}" for k                         v in metrics.items())                                                    logger.info(f"Step {{self._step}}: {{metric_str}}")