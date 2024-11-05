"""Enhanced transformer layer implementations."""

from typing import Optional, Dict, Any
import jax


class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with advanced features."""

    config: Dict[str, Any]

    def setup(self):
        """Initialize layer components."""
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.config["num_attention_heads"],
            dropout_rate=self.config["attention_dropout_rate"],
        )

        self.mlp = nn.Dense(
            features=self.config["intermediate_size"],
            kernel_init=jax.nn.initializers.normal(0.02),
        )

        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.config["dropout_rate"])

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass of the layer.

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            deterministic: Whether to use deterministic behavior
            output_attentions: Whether to output attention weights

        Returns:
            Dictionary containing layer outputs
        """
        # Self attention
        normed_hidden_states = self.layer_norm1(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            normed_hidden_states,
            mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        hidden_states = hidden_states + self.dropout(
            attention_output["hidden_states"], deterministic=deterministic
        )

        # MLP
        normed_hidden_states = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(
            mlp_output, deterministic=deterministic
        )

        outputs = {"hidden_states": hidden_states}
        if output_attentions:
            outputs["attentions"] = attention_output["attentions"]

        return outputs
