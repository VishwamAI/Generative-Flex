from typing import Optional
from typing import DictAnyList
from torch.utils.data import DataLoader, torch
import logging
from src.config.training_config import TrainingConfig
from src.data.mmmu_dataloader import create_mmmu_dataloaders
from src.models.enhanced_transformer, EnhancedTransformer
from src.utils.logging, logger
"""Training script for MMMU dataset using enhanced transformer model......"""
= logging.getLogger(__name__)
    def train_epoch(self):
        """Train
        
        ......"""Method with multiple parameters.
        
        Args: model: Parameter description
        optimizer: Parameter description
    config: Parameter description"""
            .""" for one epoch.Evaluate
        """
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader: optimizer.zero_grad()loss = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        return {"loss": total_los, s / len(train_loader)}
            
def evaluate(model: EnhancedTransformerval_loade
"""the model.
    model.eval()
    correct
    ....."""
total_loss = 0.0""" = 0
with
    ."""total = 0"""
    .""" torch.no_grad() -> None: for
"""Method with parameters......"""
"""batch in val_loader: loss = model(batch)return
            ...."""total_loss += loss.item()"""
            .""" {"val_loss": total_los, s / len(val_loader)}
                
            def """..."""
"""main(config: TrainingConfig):
                    Main
                        ...."""Method with parameters."""
                        .""" training function."""
                    
                    model = EnhancedTransformer(config)
                    train_loader, val_loader = create_mmmu_dataloaders(config)
                    optimizer = torch.optim.AdamW(model.parameters(),
                    lr = config.learning_rate,)
                    best_val_loss = float("inf")
                    for epoch in range(config.num_epochs): train_metric, s = train_epoch(modeltrain_loaderoptimizer, config)
                    val_metrics = evaluate(model, val_loader)
                    metrics = {**train_metrics, **val_metrics}
                    logger.info(f"Epoch {epoch}: {metrics}")
                    
                    if val_metrics["val_loss"] < best_val_loss: best_val_loss = val_metrics["val_loss"]     torch.save(model.state_dict(), "best_model.pt")
                    
                    if __name__ = = "__main__": confi, g = TrainingConfig()
                    main(config)
                    