import torch
import torch.nn as nn
import torch.nn.functional as F
from ..multimodal.base_transformer import BaseTransformer

class ModalityProjection(nn.Module):
    def __init__(self, config, modality_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, modality_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(modality_dim)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        return self.layer_norm(x)

class Text2XPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = BaseTransformer(config)

        # Modality-specific output projections
        self.modality_projections = nn.ModuleDict({
            'text': ModalityProjection(config, config.vocab_size),
            'image': nn.Sequential(
                ModalityProjection(config, 3072),  # 3 * 32 * 32
                nn.Unflatten(-1, (3, 32, 32)),
                nn.Upsample(scale_factor=2)
            ),
            'audio': ModalityProjection(config, config.audio_dim),
            'video': nn.Sequential(
                ModalityProjection(config, 3072),  # 3 * 32 * 32
                nn.Unflatten(-1, (3, 32, 32)),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(3, 3, 3, padding=1)
            ),
            'protein': ModalityProjection(config, 20),  # 20 amino acids
            'dna': ModalityProjection(config, 4),  # 4 nucleotides
            'music': ModalityProjection(config, config.music_dim)
        })

        # Modality-specific embeddings for conditioning
        self.modality_embeddings = nn.Embedding(len(self.modality_projections), config.hidden_size)

    def get_modality_embedding(self, modality):
        modality_idx = list(self.modality_projections.keys()).index(modality)
        return self.modality_embeddings(torch.tensor(modality_idx, device=self.transformer.embedding.weight.device))

    def forward(self, input_ids, attention_mask=None, target_modality='text', position_ids=None):
        # Add modality embedding to input embeddings
        modality_embedding = self.get_modality_embedding(target_modality)

        # Get transformer outputs
        hidden_states = self.transformer(input_ids, attention_mask, position_ids)

        # Add modality embedding to each position
        hidden_states = hidden_states + modality_embedding.unsqueeze(1)

        # Project to target modality
        if target_modality not in self.modality_projections:
            raise ValueError(f"Unsupported modality: {target_modality}")

        output = self.modality_projections[target_modality](hidden_states)

        return {
            'output': output,
            'hidden_states': hidden_states
        }

    def generate(self, input_ids, attention_mask=None, target_modality='text', max_length=None, temperature=1.0):
        if max_length is None:
            max_length = self.config.max_position_embeddings

        device = input_ids.device
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, target_modality)

            if target_modality == 'text':
                # Text generation with sampling
                logits = outputs['output'][:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                return next_token
            else:
                # Direct generation for other modalities
                return outputs['output']

    @staticmethod
    def create_attention_mask(input_ids, padding_idx=0):
        """Create attention mask from input_ids."""
        return (input_ids != padding_idx).float().unsqueeze(1).unsqueeze(2)

    def clear_cache(self):
        """Clear the transformer's attention cache."""
        self.transformer.clear_cache()
