import json
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from src.models.language_model import LanguageModel
import numpy as np
from typing import Dict, List


def load_data(
    file_path: str = "data/chatbot/training_data_cot.json",
) -> List[Dict[str, str]]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["conversations"]


def create_vocabulary(conversations: List[Dict[str, str]]) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    for conv in conversations:
        for text in [conv["input"], conv["response"]]:
            for token in text.lower().split():
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def tokenize(text: str, vocab: Dict[str, int], max_length: int) -> np.ndarray:
    tokens = ["<start>"] + text.lower().split() + ["<end>"]
    token_ids = [vocab[token] for token in tokens]
    if len(token_ids) < max_length:
        token_ids += [vocab["<pad>"]] * (max_length - len(token_ids))
    return np.array(token_ids[:max_length])


def prepare_batch(
    conversations: List[Dict[str, str]],
    vocab: Dict[str, int],
    batch_size: int,
    max_length: int,
) -> tuple:
    inputs = []
    targets = []

    for conv in conversations:
        input_ids = tokenize(conv["input"], vocab, max_length)
        target_ids = tokenize(conv["response"], vocab, max_length)
        inputs.append(input_ids)
        targets.append(target_ids)

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def create_train_state(model, learning_rate: float):
    params = model.init(
        jax.random.PRNGKey(0), jnp.ones((1, 32), dtype=jnp.int32), training=False
    )
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state, inputs, targets, rng):
    def loss_fn(params):
        logits = state.apply_fn(params, inputs, training=True, rngs={"dropout": rng})
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    # Load and prepare data
    conversations = load_data("data/chatbot/training_data.json")
    vocab = create_vocabulary(conversations)

    # Model parameters
    max_length = 32
    vocab_size = len(vocab)
    hidden_dim = 64
    num_heads = 4
    head_dim = 16
    mlp_dim = 256
    num_layers = 2
    dropout_rate = 0.1

    # Create and initialize model
    model = LanguageModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        max_seq_len=max_length,
    )

    # Prepare training data
    inputs, targets = prepare_batch(
        conversations, vocab, batch_size=len(conversations), max_length=max_length
    )

    # Initialize training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, learning_rate=1e-3)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        rng, train_rng = jax.random.split(rng)
        state, loss = train_step(
            state, jnp.array(inputs), jnp.array(targets), train_rng
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss}")

    print("Training completed!")

    # Save vocabulary
    with open("data/chatbot/vocab.json", "w") as f:
        json.dump(vocab, f)

    # Save model parameters
    with open("model_params.json", "w") as f:
        json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), state.params), f)

    print("Model parameters and vocabulary saved successfully!")


if __name__ == "__main__":
    main()
