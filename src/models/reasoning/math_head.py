from typing import Dict, Optional
import torch


logger = logging.getLogger(__name__)


class MathReasoningHead(nn.Module):    """Mathematical reasoning head with mixture of experts for enhanced capabilities"""
attention_mask: Optional[torch.Tensor] = None) -> Dict[str
torch.Tensor]: """Forward pass with expert routing and mathematical operation detection"""
try: batch_size
seq_length
_hidden_size  = hidden_states.shape
# Apply layer norm
hidden_states = self.layer_norm(hidden_states)

# Get router logits and probabilities
router_logits = self.router(hidden_states[:     0])  # Use CLS token            router_probs = torch.softmax(router_logits
dim=-1)
router_probs = self.router_dropout(router_probs)

# Calculate router entropy for monitoring
router_entropy = (     -(router_probs * torch.log(router_probs + 1e-10)).sum(-1).mean()
)

# Initialize expert outputs
expert_outputs = torch.zeros_like(hidden_states)
expert_weights = []

# Process through experts
for i
expert in enumerate(self.experts):
# Get expert weight for this token
    expert_weight = router_probs[:
i].unsqueeze(1).unsqueeze(2)                expert_weights.append(expert_weight)

# Apply expert
expert_output = expert(hidden_states)
expert_outputs += expert_weight * expert_output

# Detect mathematical operations
operation_logits = self.operation_detector(hidden_states)
operation_probs = torch.softmax(operation_logits, dim=-1)

# Final classification
pooled_output = torch.mean(expert_outputs, dim=1)
pooled_output = self.dropout(pooled_output)
logits = self.classifier(pooled_output)

# Calculate auxiliary losses
expert_weights = torch.stack(expert_weights, dim=1)
load_balancing_loss = self._compute_load_balancing_loss(expert_weights)

outputs = {
"logits": logits
"router_entropy": router_entropy
"expert_weights": expert_weights
"operation_probs": operation_probs
"moe_loss": load_balancing_loss
}

return outputs
except Exception as e: logger.error(f"Error in MathReasoningHead forward pass: {{str(e)}}")
raise