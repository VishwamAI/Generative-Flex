import json
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


class SimpleGreetingModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.hidden_size
        )
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = jnp.mean(x, axis=0)
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = self.output(x)
        return x


def create_minimal_data():
    """Create minimal training data with chain-of-thought reasoning."""
    data = {
        "conversations": [
            {
                "input": "hi",
                "thought": (
                    "1. Recognize greeting\n"
                    "2. Prepare polite response\n"
                    "3. Offer assistance"
                ),
                "response": "Hello! How can I assist you today?",
            }
        ]
    }

    # Save the training data
    with open("data/chatbot/minimal_cot_data.json", "w") as f:
        json.dump(data, f, indent=2)

    # Create vocabulary from the data
    vocab = set()
    for conv in data["conversations"]:
        vocab.update(conv["input"].split())
        vocab.update(conv["thought"].split())
        vocab.update(conv["response"].split())

    # Add special tokens
    vocab = ["<pad>", "<unk>"] + sorted(list(vocab))

    # Save vocabulary
    with open("data/chatbot/minimal_vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    return data, vocab


def main():
    print("\nCreating minimal training data with chain-of-thought...")
    data, vocab = create_minimal_data()

    # Create token mappings
    word_to_id = {word: i for i, word in enumerate(vocab)}

    # Initialize model and optimizer
    model = SimpleGreetingModel(vocab_size=len(vocab))
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.array([0])  # Single token input
    params = model.init(key, dummy_input)
    opt_state = optimizer.init(params)

    print("\nStarting training...")
    for epoch in range(100):
        # Convert input to tokens
        for conv in data["conversations"]:
            input_tokens = jnp.array(
                [word_to_id.get(w, word_to_id["<unk>"]) for w in conv["input"].split()]
            )
            target_tokens = jnp.array(
                [
                    word_to_id.get(w, word_to_id["<unk>"])
                    for w in conv["response"].split()
                ]
            )

            # Define loss function for gradient computation
            def loss_fn(params):
                logits = model.apply(params, input_tokens)
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits[None, :], target_tokens[0:1]
                ).mean()
                return loss

            # Compute gradients and update parameters
            loss_value = loss_fn(params)
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value}")

    print("\nTraining completed!")

    # Save the trained parameters
    params_dict = jax.tree_util.tree_map(lambda x: x.tolist(), params)
    with open("model_params_minimal.json", "w") as f:
        json.dump(params_dict, f)

    print("Model parameters saved to 'model_params_minimal.json'")


if __name__ == "__main__":
    main()
