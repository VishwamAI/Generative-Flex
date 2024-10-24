import json
import jax
import jax.numpy as jnp
from flax import linen as nn


# Simple model definition
class SimpleLanguageModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.hidden_size)
        self.dense = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.vocab_size)

    def __call__(self, x, training=False):
        x = self.embedding(x)
        x = self.dense(x)
        x = nn.relu(x)
        x = self.output(x)
        return x


def main():
    print("\nTesting model responses:")
    print("-" * 40)

    # Load vocabulary
    with open("data/chatbot/vocab.json", "r") as f:
        vocab = json.load(f)

    # Create token mappings
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for i, word in enumerate(vocab)}

    # Initialize model
    model = SimpleLanguageModel(vocab_size=len(vocab))

    # Load parameters
    with open("model_params.json", "r") as f:
        params_dict = json.load(f)

    # Convert parameters back to arrays
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params_dict)

    # Test input
    test_input = "hi"
    print(f"Input: {test_input}")

    # Tokenize input
    input_tokens = [
        word_to_id.get(word, word_to_id["<unk>"]) for word in test_input.split()
    ]
    input_array = jnp.array([input_tokens])

    # Generate response
    output_logits = model.apply({"params": params}, input_array)
    output_tokens = jnp.argmax(output_logits, axis=-1)

    # Convert tokens back to words
    response = " ".join([id_to_word[int(token)] for token in output_tokens[0]])
    print(f"Response: {response}")
    print("-" * 40)


if __name__ == "__main__":
    main()
