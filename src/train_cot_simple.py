import json
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


# Simple model for chain-of-thought demonstration
class SimpleCoTModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.hidden_size)
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.vocab_size)

    def __call__(self, x, training=False):
        x = self.embedding(x)
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x


def main():
    # Create minimal training data with chain-of-thought
    training_data = {
        "conversations": [
            {
                "input": "hi",
                "response": (
                    "Step 1: Acknowledge greeting. "
                    "Step 2: Offer help. "
                    "Hello! How can I assist you today?"
                ),
            }
        ]
    }

    # Save training data and create vocabulary
    with open("data/chatbot/training_data_cot.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # Create and save vocabulary
    words = set(["<unk>", "<pad>"])
    for conv in training_data["conversations"]:
        words.update(conv["input"].split())
        words.update(conv["response"].split())
    vocab = sorted(list(words))

    with open("data/chatbot/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # Convert to tokens and train
    word_to_id = {word: i for i, word in enumerate(vocab)}
    input_tokens = [
        [word_to_id.get(w, word_to_id["<unk>"]) for w in conv["input"].split()]
        for conv in training_data["conversations"]
    ]
    output_tokens = [
        [word_to_id.get(w, word_to_id["<unk>"]) for w in conv["response"].split()]
        for conv in training_data["conversations"]
    ]

    # Initialize model and train
    model = SimpleCoTModel(vocab_size=len(vocab))
    optimizer = optax.adam(0.01)

    key = jax.random.PRNGKey(0)
    x = jnp.array([input_tokens[0]])
    params = model.init(key, x)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Training loop
    print("\nTraining with chain-of-thought reasoning...")
    for epoch in range(100):
        x = jnp.array([input_tokens[0]])
        y = jnp.array([output_tokens[0]])

        def loss_fn(params):
            logits = model.apply({"params": params}, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Save model parameters
    with open("model_params.json", "w") as f:
        json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), state.params), f)
    print("\nTraining completed! Model saved.")


if __name__ == "__main__":
    main()
