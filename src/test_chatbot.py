import jax
import jax.numpy as jnp
from src.models.language_model import LanguageModel
import json
from typing import Dict, List

def load_vocab(vocab_file: str) -> Dict[str, int]:
    """Load vocabulary from file."""
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    return vocab

def tokenize(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert text to tokens using vocabulary."""
    # Simple whitespace tokenization for demonstration
    words = text.lower().split()
    return [vocab.get(word, vocab['<unk>']) for word in words]

def main():
    # Load vocabulary and model parameters
    with open('data/chatbot/vocab.json', 'r') as f:
        vocab = json.load(f)

    # Model parameters (must match training)
    max_length = 32
    vocab_size = len(vocab)
    hidden_dim = 64
    num_heads = 4
    head_dim = 16
    mlp_dim = 256
    num_layers = 2
    dropout_rate = 0.1

    # Initialize model
    model = LanguageModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        max_seq_len=max_length
    )

    # Load trained parameters
    with open('model_params.json', 'r') as f:
        params = json.load(f)

    # Test greetings
    test_inputs = [
        "hello",
        "hi",
        "good morning",
        "hey",
        "greetings"
    ]

    # Generate responses
    print("\nTesting model responses:")
    print("-" * 40)

    for input_text in test_inputs:
        # Tokenize input
        tokens = tokenize(input_text, vocab)
        input_array = jnp.array([tokens])

        # Generate response
        output = model.apply(
            params,
            input_array,
            training=False
        )

        # Convert output probabilities to tokens
        predicted_tokens = jnp.argmax(output[0], axis=-1)

        # Convert tokens back to text
        rev_vocab = {v: k for k, v in vocab.items()}
        response = ' '.join([rev_vocab[int(token)] for token in predicted_tokens])

        print(f"Input: {input_text}")
        print(f"Response: {response}")
        print("-" * 40)

if __name__ == "__main__":
    main()
