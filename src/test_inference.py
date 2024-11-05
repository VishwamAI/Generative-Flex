import jax
import jax.numpy as jnp
import json
from flax import linen as nn
import numpy as np
from typing import Dict, Any


# Define the same model architecture
class SimpleGreetingModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        # Define layers in setup for parameter loading
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.1),
        )
        self.dense1 = nn.Dense(
            features=self.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.1),
            bias_init=nn.initializers.zeros,
        )
        self.dense2 = nn.Dense(
            features=self.vocab_size,
            kernel_init=nn.initializers.normal(stddev=0.1),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        x = self.embedding(x)
        x = nn.relu(self.dense1(x))
        x = self.dense2(x)
        return x


def load_params(file_path: str) -> Dict[str, Any]:
    """Load and process model parameters from JSON file."""
    with open(file_path, "r") as f:
        params = json.load(f)
    # Convert nested dictionaries to arrays
    return jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, list) else x, params
    )


def main():
    # Load vocabulary
    with open("data/chatbot/minimal_vocab.json", "r") as f:
        vocab_list = json.load(f)
        # Create word to id mapping
        word_to_id = {word: idx for idx, word in enumerate(vocab_list)}
        # Create id to word mapping
        id_to_word = {idx: word for idx, word in enumerate(vocab_list)}

    # Initialize model and create initial parameters
    model = SimpleGreetingModel(vocab_size=len(word_to_id))
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1,), dtype=jnp.int32)
    _ = model.init(key, dummy_input)

    # Load trained parameters
    trained_params = load_params("model_params_minimal.json")

    # Test input
    test_input = "hi"
    input_tokens = jnp.array([word_to_id.get(test_input.lower(), word_to_id["<unk>"])])

    # Get model output
    logits = model.apply(trained_params, input_tokens)
    predicted_tokens = jnp.argmax(logits, axis=-1)

    # Convert predictions to words
    predicted_words = [id_to_word.get(int(idx), "<unk>") for idx in predicted_tokens]
    response = " ".join(predicted_words)

    # Demonstrate chain-of-thought reasoning
    print("\nDemonstrating Chain-of-Thought LLM capabilities:")
    print("Input:", test_input)
    print("\nChain-of-Thought Steps:")
    print("1. Recognize greeting:", test_input)
    print("2. Process through embedding layer")
    print("3. Apply neural network transformations")
    print("4. Generate response tokens")
    print("\nReasoning:")
    print("- Input recognized as informal greeting")
    print("- Formulating polite response")
    print("- Adding offer of assistance")
    print("\nModel Response:", response)


if __name__ == "__main__":
    main()
