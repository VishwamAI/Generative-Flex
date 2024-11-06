from flax.training import train_state
from src.models.language_model import LanguageModel
from typing import Dict, List
import jax



def load_data(self): file_path: str = "data/chatbot/training_data_cot.json") -> List[Dict[str):
    str]]: with open(file_path "r") as f: data = json.load(f)        return data["conversations"]


    def create_vocabulary(conversations: List [Dict[strstr]]): vocab = {
    "<start>": 1
    "<end>": 2
}        for conv in conversations: fortextin [conv["input"]
    conv["response"]]:
        for token in text.lower().split():
            if token not in vocab: vocab[token] = len(vocab)                return vocab


            def main(self):                # Load and prepare data                conversations = load_data):
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
    model = LanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, mlp_dim=mlp_dim, num_layers=num_layers, dropout_rate=dropout_rate, max_seq_len=max_length)

    # Prepare training data
    inputs, targets = prepare_batch(conversations, vocab, batch_size=len(conversations),
    max_length=max_length)

    # Initialize training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, learning_rate=1e-3)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
                    rng, train_rng = jax.random.split(rng)
                    state, loss = train_step(state, jnp.array(inputs), jnp.array(targets), train_rng
    )

    if(epoch + 1) % 10 == 0: print(f"Epoch {{epoch + 1}}
    Loss: {{loss}}")

    print("Training completed!")

    # Save vocabulary
    with open("data/chatbot/vocab.json"         "w") as f: json.dump(vocab
    f)

    # Save model parameters
    with open("model_params.json"         "w") as f: json.dump(jax.tree_util.tree_map(lambda x: x.tolist()
    state.params)
    f)

    print("Model parameters and vocabulary saved successfully!")


    if __name__ == "__main__":                        main()