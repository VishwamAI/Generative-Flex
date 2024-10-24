import jax
import jax.numpy as jnp
import json
from flax import linen as nn
import numpy as np
from flax.training import train_state

class SimpleModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.hidden_size)
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = self.output(x)
        return x

def init_model_state(model, rng, vocab_size):
    """Initialize model state with dummy input."""
    dummy_input = jnp.ones((1,), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    return params

def load_params(file_path):
    """Load and process saved parameters."""
    with open(file_path, 'r') as f:
        saved_params = json.load(f)
    # Convert lists to numpy arrays recursively
    def process_value(x):
        if isinstance(x, list):
            return np.array(x)
        elif isinstance(x, dict):
            return {k: process_value(v) for k, v in x.items()}
        return x
    return process_value(saved_params)

def main():
    # Load vocabulary
    with open('data/chatbot/minimal_vocab.json', 'r') as f:
        vocab_list = json.load(f)
        word_to_id = {word: idx for idx, word in enumerate(vocab_list)}
        id_to_word = {idx: word for idx, word in enumerate(vocab_list)}

    # Initialize model and parameters
    rng = jax.random.PRNGKey(0)
    model = SimpleModel(vocab_size=len(word_to_id))
    initial_params = init_model_state(model, rng, len(word_to_id))

    # Load trained parameters
    trained_params = load_params('model_params_minimal.json')

    # Test input
    test_input = "hi"
    input_token = jnp.array([word_to_id.get(test_input.lower(), word_to_id['<unk>'])])

    # Get model output
    logits = model.apply(trained_params, input_token)
    predicted_token = jnp.argmax(logits, axis=-1)

    # Convert prediction to words
    response = ' '.join([id_to_word[int(idx)] for idx in predicted_token])

    # Demonstrate chain-of-thought
    print("\nChain-of-Thought Demonstration:")
    print("Input:", test_input)
    print("\nReasoning Steps:")
    print("1. Recognize input as greeting:", test_input)
    print("2. Convert to token ID:", int(input_token[0]))
    print("3. Process through neural network")
    print("4. Generate response token:", int(predicted_token[0]))
    print("5. Convert to text:", response)
    print("\nFinal Response:", response)

if __name__ == "__main__":
    main()
