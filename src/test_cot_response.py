import json
import jax
import jax.numpy as jnp
from flax import linen as nn

class SimpleChatModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size
        )
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = jnp.mean(x, axis=0)  # Average over sequence length
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = self.output(x)
        return x

def main():
    print("\nTesting Chain-of-Thought Response Generation:")
    print("-" * 50)

    # Load vocabulary
    with open('data/chatbot/vocab.json', 'r') as f:
        vocab = json.load(f)

    # Create token mappings
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for i, word in enumerate(vocab)}

    # Initialize model
    model = SimpleChatModel(vocab_size=len(vocab))

    # Load trained parameters
    with open('model_params.json', 'r') as f:
        params_dict = json.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params_dict)

    # Test input
    test_input = "hi"
    print(f"Input: {test_input}")

    # Tokenize input
    input_tokens = jnp.array([word_to_id.get(w, word_to_id['<unk>']) for w in test_input.split()])

    # Generate response
    logits = model.apply({'params': params}, input_tokens)
    predicted_tokens = jnp.argsort(logits)[-10:][::-1]  # Get top 10 predictions

    # Convert tokens back to words
    response_words = [id_to_word[int(token)] for token in predicted_tokens]
    response = ' '.join(response_words)
    print(f"Generated Response: {response}")
    print("-" * 50)

if __name__ == "__main__":
    main()
