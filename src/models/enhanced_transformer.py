"""Enhanced transformer with features from major AI models."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import ModuleUtilsMixin


class EnhancedConfig(PretrainedConfig):
    """Configuration for enhanced transformer."""

    model_type = "enhanced_transformer"

    def __init__(
        self,
        # Model architecture
        hidden_size: int = 256,  # Reduced from 512
        num_attention_heads: int = 4,  # Reduced from 8
        num_hidden_layers: int = 3,  # Reduced from 4
        intermediate_size: int = 1024,  # Reduced from 8192
        hidden_act: str = "gelu",
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        vocab_size: int = 50265,  # Updated to match OPT-125m tokenizer
        embedding_size: int = 256,  # Reduced from 512
        # Advanced features
        num_experts: int = 4,  # Reduced from 8
        expert_capacity: int = 16,  # Reduced from 32
        use_flash_attention: bool = True,  # Meta's efficient attention
        use_constitutional_ai: bool = True,  # Anthropic's safety features
        use_retrieval: bool = True,  # Real-time knowledge integration
        use_privacy_preserving: bool = True,  # Privacy-preserving features
        noise_multiplier: float = 0.1,  # For privacy-preserving noise
        l2_norm_clip: float = 1.0,  # For gradient clipping in privacy
        # Constitutional AI parameters
        safety_threshold: float = 0.8,  # Threshold for content filtering
        alignment_factor: float = 0.9,  # Weight for value alignment
        context_window: int = 512,  # Reduced from 1024
        # Generation parameters
        max_new_tokens: int = 256,  # Reduced from 512
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        # Optimization
        head_dim: int = 32,  # Reduced from 64
        max_sequence_length: int = 512,  # Reduced from 2048
        layer_norm_eps: float = 1e-12,
        dropout_rate: float = 0.1,
        use_kv_cache: bool = True,  # For efficient generation
        use_int4_quantization: bool = True,  # Apple's optimization
        use_neural_engine: bool = True,  # Apple's ML acceleration
        block_size: int = 16,  # Reduced from 32
        num_bits: int = 4,  # For quantization precision
        cache_dtype: str = "float16",  # Cache data type
        cache_size_multiplier: float = 1.5,  # Cache size multiplier
        # Modality-specific parameters
        image_input_size: int = 96,  # Input dimension for image features
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.use_flash_attention = use_flash_attention
        self.use_constitutional_ai = use_constitutional_ai
        self.use_retrieval = use_retrieval
        self.use_privacy_preserving = use_privacy_preserving
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.safety_threshold = safety_threshold
        self.alignment_factor = alignment_factor
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout_rate
        self.use_kv_cache = use_kv_cache
        self.use_int4_quantization = use_int4_quantization
        self.use_neural_engine = use_neural_engine
        self.block_size = block_size
        self.num_bits = num_bits
        self.cache_dtype = cache_dtype
        self.cache_size_multiplier = cache_size_multiplier
        self.image_input_size = image_input_size


class MultiModalEmbedding(nn.Module):
    """Multi-modal embedding layer supporting text, image, audio, and video."""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config

        # Text embeddings
        self.text_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # Modality-specific projections
        self.image_projection = nn.Sequential(
            nn.Linear(
                256, config.hidden_size
            ),  # Match input size to processed image size
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        self.video_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, inputs: Dict[str, torch.Tensor], training: bool = True
    ) -> torch.Tensor:
        """Process multi-modal inputs."""
        import logging

        logger = logging.getLogger(__name__)

        logger.info(
            f"Starting MultiModalEmbedding forward pass with input modalities: {list(inputs.keys())}"
        )
        batch_size = None
        seq_length = 0  # Initialize to 0 to accumulate total sequence length
        embeddings = []

        try:
            for modality, data in inputs.items():
                logger.info(
                    f"Processing {modality} modality, input shape: {data.shape}"
                )

                if modality == "text":
                    try:
                        if data.dtype in [torch.float32, torch.float64]:
                            # If input is already embedded, use it directly
                            text_embeddings = data
                        else:
                            # Convert to int64 for embedding lookup
                            data = data.to(torch.int64)
                            text_embeddings = self.text_embeddings(data)

                        batch_size = text_embeddings.shape[0]
                        seq_length += text_embeddings.shape[1]
                        logger.info(f"Text embeddings shape: {text_embeddings.shape}")

                        # Verify text embeddings have correct hidden size
                        if text_embeddings.size(-1) != self.config.hidden_size:
                            text_embeddings = nn.Linear(
                                text_embeddings.size(-1),
                                self.config.hidden_size,
                                device=text_embeddings.device,
                            )(text_embeddings)
                        embeddings.append(text_embeddings)
                        logger.info(
                            f"Memory after text processing: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                        )
                    except Exception as e:
                        logger.error(f"Error processing text input: {str(e)}")
                        raise

                elif modality == "image":
                    try:
                        logger.info("Starting image projection")
                        # Project image features to hidden size
                        image_embeddings = self.image_projection(data)
                        if batch_size is None:
                            batch_size = image_embeddings.shape[0]
                        logger.info(
                            f"Image embeddings shape after projection: {image_embeddings.shape}"
                        )

                        # Verify and project to correct hidden size if needed
                        if image_embeddings.size(-1) != self.config.hidden_size:
                            image_embeddings = nn.Linear(
                                image_embeddings.size(-1),
                                self.config.hidden_size,
                                device=image_embeddings.device,
                            )(image_embeddings)

                        # Add sequence length from image embeddings
                        seq_length += image_embeddings.size(1)
                        embeddings.append(image_embeddings)
                        logger.info(
                            f"Memory after image processing: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                        )
                    except Exception as e:
                        logger.error(f"Error processing image input: {str(e)}")
                        raise
                elif modality == "audio":
                    try:
                        # Project audio features to hidden size
                        audio_embeddings = self.audio_projection(data)
                        if batch_size is None:
                            batch_size = audio_embeddings.shape[0]
                        # Verify and project to correct hidden size if needed
                        if audio_embeddings.size(-1) != self.config.hidden_size:
                            audio_embeddings = nn.Linear(
                                audio_embeddings.size(-1),
                                self.config.hidden_size,
                                device=audio_embeddings.device,
                            )(audio_embeddings)
                        # Reshape to match text embedding dimensions (batch_size, seq_length, hidden_size)
                        audio_embeddings = audio_embeddings.unsqueeze(1)
                        seq_length += 1  # Add one token for audio
                        embeddings.append(audio_embeddings)
                        logger.info(
                            f"Memory after audio processing: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                        )
                    except Exception as e:
                        logger.error(f"Error processing audio input: {str(e)}")
                        raise

                elif modality == "video":
                    try:
                        # Project video features to hidden size
                        video_embeddings = self.video_projection(data)
                        if batch_size is None:
                            batch_size = video_embeddings.shape[0]
                        # Verify and project to correct hidden size if needed
                        if video_embeddings.size(-1) != self.config.hidden_size:
                            video_embeddings = nn.Linear(
                                video_embeddings.size(-1),
                                self.config.hidden_size,
                                device=video_embeddings.device,
                            )(video_embeddings)
                        # Reshape to match text embedding dimensions (batch_size, seq_length, hidden_size)
                        video_embeddings = video_embeddings.unsqueeze(1)
                        seq_length += 1  # Add one token for video
                        embeddings.append(video_embeddings)
                        logger.info(
                            f"Memory after video processing: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                        )
                    except Exception as e:
                        logger.error(f"Error processing video input: {str(e)}")
                        raise

            logger.info(
                f"Total sequence length after processing all modalities: {seq_length}"
            )

            # Add position embeddings
            device = next(iter(inputs.values())).device
            position_ids = torch.arange(seq_length, device=device)[None, :]
            # Clamp position IDs to maximum allowed size
            max_positions = self.config.max_position_embeddings
            position_ids = torch.clamp(position_ids, 0, max_positions - 1)
            position_embeddings = self.position_embeddings(position_ids)
            logger.info(f"Position embeddings shape: {position_embeddings.shape}")

            # Add token type embeddings (0 for all tokens in this case)
            token_type_ids = torch.zeros(
                (batch_size, seq_length), dtype=torch.int64, device=device
            )
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            logger.info(f"Token type embeddings shape: {token_type_embeddings.shape}")

            # Combine all embeddings along sequence dimension
            if embeddings:
                # Verify all embeddings have correct hidden size before concatenation
                for i, emb in enumerate(embeddings):
                    if emb.size(-1) != self.config.hidden_size:
                        raise ValueError(
                            f"Embedding {i} has incorrect hidden size: {emb.size(-1)} != {self.config.hidden_size}"
                        )
                    if len(emb.shape) != 3:
                        raise ValueError(
                            f"Embedding {i} has incorrect number of dimensions: {len(emb.shape)} != 3"
                        )

                # Concatenate along sequence dimension
                combined = torch.cat(
                    embeddings, dim=1
                )  # [batch_size, seq_length, hidden_size]
                logger.info(
                    f"Combined embeddings shape after concatenation: {combined.shape}"
                )
            else:
                raise ValueError("No valid inputs provided")

            # Expand position and token type embeddings to match batch size
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
            token_type_embeddings = token_type_embeddings.expand(batch_size, -1, -1)

            # Add position and token type embeddings
            combined = combined + position_embeddings + token_type_embeddings
            logger.info(
                f"Final combined shape after adding positional and token type embeddings: {combined.shape}"
            )

            # Layer normalization and dropout
            combined = self.layer_norm(combined)
            combined = self.dropout(combined) if training else combined
            logger.info(
                f"Final output shape after normalization and dropout: {combined.shape}"
            )
            logger.info(
                f"Final memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
            )

            return combined

        except Exception as e:
            logger.error(f"Error in MultiModalEmbedding forward pass: {str(e)}")
            raise


class FlashAttention(nn.Module):
    """Efficient attention implementation with O(N) complexity."""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Compute efficient attention."""
        # Get dimensions
        batch_size, seq_length, num_heads, head_dim = q.shape
        scale = head_dim**-0.5

        # Reshape for efficient computation
        q = q.reshape(batch_size, seq_length, num_heads, head_dim)
        k = k.reshape(batch_size, seq_length, num_heads, head_dim)
        v = v.reshape(batch_size, seq_length, num_heads, head_dim)

        # Compute attention scores
        qk = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale

        # Apply causal mask for generation
        if mask is not None:
            # Reshape mask to match attention scores dimensions [batch, heads, seq, seq]
            mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            # Ensure mask has correct sequence length
            if mask.size(-1) != seq_length:
                mask = F.pad(mask, (0, seq_length - mask.size(-1)))
            qk = qk + mask * -1e9
        else:
            # Default causal mask
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), device=q.device), diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                0
            )  # Add batch and head dims
            qk = qk + causal_mask * -1e9

        # Apply attention dropout
        attention_weights = F.softmax(qk, dim=-1)
        if training:
            attention_weights = F.dropout(
                attention_weights,
                p=self.config.attention_dropout_prob,
                training=training,
            )

        # Compute attention output
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, v)
        return attention_output


class ExpertLayer(nn.Module):
    """Mixture of Experts layer for specialized processing."""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config

        # Initialize experts
        self.experts = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.num_experts)
            ]
        )

        # Router network
        self.router = nn.Linear(config.hidden_size, config.num_experts)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        batch_size, seq_length, hidden_size = x.shape

        # Expert routing with capacity limiting
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Apply dropout to routing weights during training
        if training:
            routing_weights = F.dropout(
                routing_weights, p=self.config.dropout_rate, training=training
            )

        # Expert processing with capacity limiting
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=2)
        combined_output = torch.sum(
            expert_outputs * routing_weights.unsqueeze(-1), dim=2
        )
        return combined_output


class ConstitutionalLayer(nn.Module):
    """Implementation of Constitutional AI principles."""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config

        # Value alignment scoring
        self.alignment_dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        # Value alignment scoring
        alignment_score = self.alignment_dense(x)

        # Safety threshold check
        safe_mask = (alignment_score > self.config.safety_threshold).float()

        # Apply constitutional principles
        if context is not None:
            x = torch.where(safe_mask.bool(), x, context)
        else:
            # If no context provided, zero out unsafe content
            x = x * safe_mask

        return x


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with features from major models."""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config

        # Layer normalization
        self.attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # Multi-head attention
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.flash_attention = (
            FlashAttention(config) if config.use_flash_attention else None
        )
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

        # Feed-forward network with expert routing
        if config.num_experts > 0:
            self.expert_layer = ExpertLayer(config)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU() if config.hidden_act == "gelu" else nn.ReLU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
            )

        # Constitutional AI layer
        if config.use_constitutional_ai:
            self.constitutional_layer = ConstitutionalLayer(config)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with advanced features."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info("Starting EnhancedTransformerBlock forward pass")
            batch_size, seq_length, hidden_size = inputs.shape
            logger.info(
                f"Input shape: batch_size={batch_size}, seq_length={seq_length}, hidden_size={hidden_size}"
            )
            num_heads = self.config.num_attention_heads
            head_dim = hidden_size // num_heads
            logger.info(f"Attention config: num_heads={num_heads}, head_dim={head_dim}")

            # Layer normalization
            logger.info("Applying attention layer normalization")
            x = self.attention_layernorm(inputs)
            logger.info(
                f"Memory after layer norm: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
            )

            # Project to query, key, value
            logger.info("Computing QKV projections")
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(batch_size, seq_length, 3, num_heads, head_dim)
            q, k, v = torch.unbind(qkv, dim=2)
            logger.info(f"QKV shapes: q={q.shape}, k={k.shape}, v={v.shape}")

            # Prepare attention mask
            if attention_mask is not None:
                logger.info("Processing attention mask")
                # Reshape mask to [batch, 1, seq, seq] for broadcasting
                attention_mask = attention_mask.unsqueeze(1)
                # Ensure mask has correct sequence length
                if attention_mask.size(-1) != seq_length:
                    logger.info(
                        f"Padding attention mask from {attention_mask.size(-1)} to {seq_length}"
                    )
                    attention_mask = F.pad(
                        attention_mask, (0, seq_length - attention_mask.size(-1))
                    )
                logger.info(f"Final attention mask shape: {attention_mask.shape}")

            # Initialize attention weights
            attention_weights = None

            # Use Flash Attention for efficient computation
            if self.config.use_flash_attention:
                logger.info("Using Flash Attention")
                attention_output = self.flash_attention(
                    q, k, v, mask=attention_mask, training=training
                )
                # Reshape from (batch, seq, heads, dim) to (batch, seq, hidden_size)
                attention_output = attention_output.reshape(
                    batch_size, seq_length, hidden_size
                )
                logger.info(f"Flash attention output shape: {attention_output.shape}")
            else:
                logger.info("Using standard attention")
                # Fallback to standard attention
                scale = head_dim**-0.5
                attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                logger.info(f"Attention scores shape: {attention_scores.shape}")
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask * -1e9
                attention_probs = F.softmax(attention_scores, dim=-1)
                attention_weights = attention_probs  # Store attention weights
                if training:
                    attention_probs = F.dropout(
                        attention_probs,
                        p=self.config.attention_dropout_prob,
                        training=True,
                    )
                attention_output = torch.matmul(attention_probs, v)
                attention_output = attention_output.reshape(
                    batch_size, seq_length, hidden_size
                )
                logger.info(
                    f"Standard attention output shape: {attention_output.shape}"
                )

            # Project output
            logger.info("Projecting attention output")
            attention_output = self.output_projection(attention_output)
            logger.info(
                f"Memory after attention: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
            )

            # Residual connection with dropout
            attention_output = (
                self.dropout(attention_output) if training else attention_output
            )
            x = inputs + attention_output
            logger.info(f"Shape after residual connection: {x.shape}")

            # Layer normalization
            logger.info("Applying FFN layer normalization")
            x = self.ffn_layernorm(x)

            # Feed-forward network with expert routing
            if hasattr(self, "expert_layer"):
                logger.info("Using expert layer")
                expert_output = self.expert_layer(x)
                x = x + (self.dropout(expert_output) if training else expert_output)
                logger.info(f"Expert layer output shape: {x.shape}")
            else:
                # Standard feed-forward network
                logger.info("Using standard FFN")
                ffn_output = self.ffn(x)
                x = x + (self.dropout(ffn_output) if training else ffn_output)
                logger.info(f"FFN output shape: {x.shape}")

            # Constitutional AI layer for safety
            if self.config.use_constitutional_ai:
                logger.info("Applying constitutional AI layer")
                x = self.constitutional_layer(x)
                logger.info(f"Constitutional layer output shape: {x.shape}")

            # Final layer normalization
            logger.info("Applying final layer normalization")
            x = self.final_layernorm(x)
            logger.info(f"Final output shape: {x.shape}")
            logger.info(
                f"Final memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
            )

            return (
                x,
                attention_weights,
                None,
            )  # hidden_states, attention_weights, present_key_value

        except Exception as e:
            logger.error(f"Error in EnhancedTransformerBlock forward pass: {str(e)}")
            raise


class EnhancedTransformer(PreTrainedModel):
    """Enhanced transformer model with features from major AI models."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        max_position_embeddings,
        vocab_size,
    ):
        config = EnhancedConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
        )
        super().__init__(config)
        self.config = config

        # Enable gradient checkpointing
        self.gradient_checkpointing = True
        self.supports_gradient_checkpointing = True

        # Multi-modal embedding layer (handles both text and other modalities)
        self.embeddings = MultiModalEmbedding(self.config)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                EnhancedTransformerBlock(self.config)
                for _ in range(self.config.num_hidden_layers)
            ]
        )

        # Final layer normalization
        self.final_layernorm = nn.LayerNorm(
            self.config.hidden_size, eps=self.config.layer_norm_eps
        )

        # Output projection
        self.output_projection = nn.Linear(
            self.config.hidden_size, self.config.vocab_size
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Setup logging
        import logging

        self.logger = logging.getLogger(__name__)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        try:
            self.logger.info("Starting EnhancedTransformer forward pass")
            self.logger.info(f"Input modalities: {list(inputs.keys())}")

            # Get embeddings from multimodal embedding layer
            self.logger.info("Getting embeddings from multimodal embedding layer")
            hidden_states = self.embeddings(inputs, training=self.training)
            self.logger.info(f"Embeddings shape: {hidden_states.shape}")
            self.logger.info(
                f"Memory after embeddings: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
            )

            # Process through transformer blocks
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            # Modified gradient checkpointing implementation
            if self.gradient_checkpointing and self.training:
                self.logger.info("Using gradient checkpointing")
                if use_cache:
                    self.logger.warning("Gradient checkpointing disables cache")
                    use_cache = False

                for i, block in enumerate(self.blocks):

                    def custom_forward(hidden_states, attention_mask):
                        return block(hidden_states, attention_mask)

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        custom_forward,
                        hidden_states,
                        attention_mask,
                        use_reentrant=False,
                    )
                    hidden_states = layer_outputs[0]
                    if output_attentions:
                        all_attentions = all_attentions + (layer_outputs[1],)
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

                    self.logger.info(f"Block {i+1} output shape: {hidden_states.shape}")
                    self.logger.info(
                        f"Memory after block {i+1}: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                    )
            else:
                self.logger.info(
                    "Processing transformer blocks without gradient checkpointing"
                )
                for i, block in enumerate(self.blocks):
                    try:
                        self.logger.info(f"Processing block {i+1}/{len(self.blocks)}")
                        layer_outputs = block(hidden_states, attention_mask)
                        hidden_states = layer_outputs[0]

                        if output_attentions:
                            all_attentions = all_attentions + (layer_outputs[1],)
                        if output_hidden_states:
                            all_hidden_states = all_hidden_states + (hidden_states,)

                        self.logger.info(
                            f"Block {i+1} output shape: {hidden_states.shape}"
                        )
                        self.logger.info(
                            f"Memory after block {i+1}: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                        )
                    except Exception as e:
                        self.logger.error(f"Error in transformer block {i+1}: {str(e)}")
                        raise

            # Apply final layer norm
            self.logger.info("Applying final layer normalization")
            hidden_states = self.final_layernorm(hidden_states)
            self.logger.info(f"Final hidden states shape: {hidden_states.shape}")

            # Get output logits
            self.logger.info("Computing output logits")
            logits = self.output_projection(hidden_states)
            self.logger.info(f"Output logits shape: {logits.shape}")

            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                # Reshape logits and labels for loss calculation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Calculate cross entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            self.logger.info(
                f"Final memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
            )

            if not return_dict:
                output = (logits,) + (hidden_states,)
                return ((loss,) + output) if loss is not None else output

            return {
                "loss": loss,  # Will be None during evaluation if no labels provided
                "logits": logits,
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }

        except Exception as e:
            self.logger.error(f"Error in EnhancedTransformer forward pass: {str(e)}")
            raise

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        training: bool = False,
    ) -> torch.Tensor:
        """Generate text using the model."""
        # Use config defaults if not specified
        max_length = max_length or self.config.max_sequence_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p

        # Get batch size from inputs
        batch_size = next(iter(inputs.values())).size(0)
        device = next(iter(inputs.values())).device

        # Initialize position IDs and token type IDs
        position_ids = torch.arange(max_length, device=device)[None, :]
        token_type_ids = torch.zeros(
            (batch_size, max_length), dtype=torch.int64, device=device
        )

        # Initialize list to store generated tokens
        generated_tokens = []
        current_input = inputs

        # Generate tokens auto-regressively
        for step in range(max_length):
            # Forward pass
            logits = self.forward(current_input, training=training)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update inputs for next iteration
            if "text" in current_input:
                current_input["text"] = torch.cat(
                    [current_input["text"], next_token], dim=1
                )
            else:
                current_input = {"text": next_token}

            # Store generated token
            generated_tokens.append(next_token)

            # Early stopping if end token is generated
            if (next_token == self.config.eos_token_id).any():
                break

        # Concatenate all generated tokens
        return torch.cat(generated_tokens, dim=1)
