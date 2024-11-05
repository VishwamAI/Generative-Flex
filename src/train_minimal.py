import json
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


# Simple model definition (same as in test_minimal.py)
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


def create_vocab(text):
    # Create vocabulary from text
    words = set()
    words.add("<unk>")  # Unknown token
    words.add("<pad>")  # Padding token
    for sentence in text:
        words.update(sentence.split())
    return sorted(list(words))


def main():
    # Load training data
    with open("data/chatbot/training_data_minimal.json", "r") as f:
        data = json.load(f)

    # Prepare training examples
    input_text = [conv["input"] for conv in data["conversations"]]
    output_text = [conv["response"] for conv in data["conversations"]]

    # Create vocabulary
    all_text = input_text + output_text
    vocab = create_vocab(all_text)
    word_to_id = {word: i for i, word in enumerate(vocab)}

    # Save vocabulary
    with open("data/chatbot/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # Convert text to tokens
    input_tokens = [
        [word_to_id.get(word, word_to_id["<unk>"]) for word in text.split()]
        for text in input_text
    ]
    output_tokens = [
        [word_to_id.get(word, word_to_id["<unk>"]) for word in text.split()]
        for text in output_text
    ]

    # Initialize model and optimizer
    model = SimpleLanguageModel(vocab_size=len(vocab))
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 5), dtype=jnp.int32)
    params = model.init(key, dummy_input)

    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for i in range(len(input_tokens)):
            x = jnp.array([input_tokens[i]])
            y = jnp.array([output_tokens[i]])

            def loss_fn(params):
                logits = model.apply(params, x)
                return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    print("Training completed!")

    # Save model parameters
    with open("model_params.json", "w") as f:
        json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), state.params), f)

    print("Model parameters and vocabulary saved successfully!")


if __name__ == "__main__":
    main()
