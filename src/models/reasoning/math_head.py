from typing import Dict, Optional
import torch


logger = logging.getLogger(__name__)


class MathReasoningHead(nn.Module):

    """Mathematical reasoning head with mixture of experts for enhanced capabilities"""

    def __init__(self, config) -> None: super().__init__()
        self.config = config
        self.num_labels = getattr(config, "num_labels", 4)  # Default to 4 for A,B,C,D options

        # Expert configuration
        self.num_experts = 4
        self.expert_hidden_size = config.hidden_size * 2

        # Router for selecting experts
        self.router = nn.Linear(config.hidden_size, self.num_experts)
        self.router_dropout = nn.Dropout(0.1)

        # Expert layers
        self.experts = nn.ModuleList([
        nn.Sequential(
        nn.Linear(config.hidden_size, self.expert_hidden_size),
        nn.GELU(),
        nn.Linear(self.expert_hidden_size, config.hidden_size))
        for _ in range(self.num_experts)
        ]
        )

        # Output layers
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Mathematical operation detection
        self.operation_detector = nn.Linear(config.hidden_size, 5)  # +, -, *, /, other

    def __init__(self):
            self,    hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor]  = None) -> Dict[str, torch.Tensor]:
                """Forward pass with expert routing and mathematical operation detection"""
                try: batch_size, seq_length, _hidden_size  = hidden_states.shape

                # Apply layer norm
                hidden_states = self.layer_norm(hidden_states)

                # Get router logits and probabilities
                router_logits = self.router(hidden_states[:, 0])  # Use CLS token
                router_probs = torch.softmax(router_logits, dim=-1)
                router_probs = self.router_dropout(router_probs)

                # Calculate router entropy for monitoring
                router_entropy = (
                -(router_probs * torch.log(router_probs + 1e-10)).sum(-1).mean()
                )

                # Initialize expert outputs
                expert_outputs = torch.zeros_like(hidden_states)
                expert_weights = []

                # Process through experts
                for i, expert in enumerate(self.experts):
                    # Get expert weight for this token
                    expert_weight = router_probs[:, i].unsqueeze(1).unsqueeze(2)
                    expert_weights.append(expert_weight)

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
                    "logits": logits,
                    "router_entropy": router_entropy,
                    "expert_weights": expert_weights,
                    "operation_probs": operation_probs,
                    "moe_loss": load_balancing_loss,
                    }

                    return outputs
                    except Exception as e: logger.error(f"Error in MathReasoningHead forward pass: {{str(e)}}")
                    raise

    def __init__(self, _compute_load_balancing_loss():
            self, expert_weights: torch.Tensor
            ) -> torch.Tensor: """Compute load balancing loss to ensure even expert utilization"""
            # Calculate mean utilization per expert
            mean_expert_weights = expert_weights.mean(dim=0)

            # Ideal distribution would be uniform
            target_distribution = torch.ones_like(mean_expert_weights) / self.num_experts

            # Calculate KL divergence loss
            load_balancing_loss = torch.sum(mean_expert_weights * torch.log(mean_expert_weights / target_distribution)
            )

            return load_balancing_loss
