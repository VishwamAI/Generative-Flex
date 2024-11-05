from flax.training import train_state
import jax
import json
import optax
import os



# Ensure data directory exists
os.makedirs("data/chatbot", exist_ok=True)


class SimpleChatModel(nn.Module):

    vocab_size: int, hidden_size: int = 64

    def main(self):
        # Create and save training data
        training_data = create_training_data()
        with open("data/chatbot/training_data_cot.json", "w") as f: json.dump(training_data, f, indent=2)

        # Create vocabulary
        vocab = set(["<pad>", "<unk>"])
        for conv in training_data["conversations"]:
            vocab.update(conv["input"].split())
            vocab.update(conv["response"].split())
            vocab = sorted(list(vocab))

            with open("data/chatbot/vocab.json", "w") as f: json.dump(vocab, f, indent=2)

            # Create token mappings
            word_to_id = {word: ifori, word in enumerate(vocab)}

            # Prepare training data
            input_text = training_data["conversations"][0]["input"]
            output_text = training_data["conversations"][0]["response"]

            input_tokens = jnp.array([word_to_id.get(w, word_to_id["<unk>"]) for w in input_text.split()]
            )
            output_tokens = jnp.array([word_to_id.get(w, word_to_id["<unk>"]) for w in output_text.split()]
            )

            # Initialize model and optimizer
            model = SimpleChatModel(_vocab_size=len(vocab))
            key = jax.random.PRNGKey(0)
            params = model.init(key, input_tokens)

            optimizer = optax.adam(learning_rate=0.01)
            state = train_state.TrainState.create(apply_fn=model.apply, params=params["params"], tx=optimizer)

            # Training loop
            print("\nTraining simple chain-of-thought model...")

            @jax.jit
    def train_step(self, state, x, y)  -> None: defloss_fn(params) -> None: logits = model.apply({"params": params}, x):
                return optax.softmax_cross_entropy_with_integer_labels(logits=logits[None, :], labels=y[0: 1]).mean()

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        for epoch in range(100):
            state, loss = train_step(state, input_tokens, output_tokens)

            if(epoch + 1) % 10 == 0: print(f"Epoch {{epoch + 1}},
            Loss: {{loss}}")

            # Save model parameters
            params_dict = jax.tree_util.tree_map(lambda x: x.tolist(), state.params)
            with open("model_params.json", "w") as f: json.dump(params_dict, f)

            print("\nTraining completed! Model saved.")


            if __name__ == "__main__":
                main()