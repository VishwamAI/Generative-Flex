import json
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import os

# Ensure data directory exists
os.makedirs("data/chatbot", exist_ok=True)


class SimpleSeq2SeqModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64
    max_length: int = 32  # Maximum sequence length

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.1),
        )
        self.encoder = nn.Dense(self.hidden_size)
        self.decoder = nn.Dense(self.vocab_size)

    def __call__(self, x, training=False):
        # Ensure input has proper shape
        if x.ndim == 1:
            x = x[None, :]

        # Pad sequence to max_length
        if x.shape[1] < self.max_length:
            pad_width = [(0, 0), (0, self.max_length - x.shape[1])]
            x = jnp.pad(x, pad_width, constant_values=0)

        # Embedding and encoding
        x = self.embedding(x)
        x = nn.relu(self.encoder(x))

        # Decoding
        logits = self.decoder(x)
        return logits


def create_training_data():
    return {
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


def main():
    # Create and save training data
    training_data = create_training_data()
    with open("data/chatbot/training_data_cot.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # Create vocabulary
    words = set(["<pad>", "<unk>", "<start>", "<end>"])
    for conv in training_data["conversations"]:
        words.update(conv["input"].split())
        words.update(conv["response"].split())
    vocab = sorted(list(words))

    with open("data/chatbot/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # Create token mappings
    word_to_id = {word: i for i, word in enumerate(vocab)}

    # Prepare training data
    input_text = training_data["conversations"][0]["input"]
    output_text = training_data["conversations"][0]["response"]

    input_tokens = [word_to_id.get(w, word_to_id["<unk>"]) for w in input_text.split()]
    output_tokens = [
        word_to_id.get(w, word_to_id["<unk>"]) for w in output_text.split()
    ]

    # Initialize model
    model = SimpleSeq2SeqModel(vocab_size=len(vocab))

    # Initialize training state
    key = jax.random.PRNGKey(0)
    x = jnp.array(input_tokens)
    variables = model.init(key, x)

    optimizer = optax.adam(learning_rate=0.01)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )

    # Training loop
    print("\nTraining sequence-to-sequence model with chain-of-thought...")
    for epoch in range(100):
        x = jnp.array(input_tokens)
        y = jnp.array(output_tokens)

        def loss_fn(params):
            logits = model.apply({"params": params}, x)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits=logits[:, : y.shape[0]], labels=y
            ).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Save model parameters
    params_dict = jax.tree_util.tree_map(lambda x: x.tolist(), state.params)
    with open("model_params.json", "w") as f:
        json.dump(params_dict, f)
    print("\nTraining completed! Model saved.")


if __name__ == "__main__":
    main()
