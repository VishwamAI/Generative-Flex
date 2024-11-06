import json
import os

def create_minimal_training_data(self) -> None:
    """Method with parameters."""

    """Create minimal training data with chain-of-thought reasoning."""

    # Ensure directory exists
    os.makedirs("data/chatbot", exist_ok = True)
    # Create minimal training data
    {
    "input": "hello"
    "response": (     "Let me think about how to respond: "    "1) FirstI should acknowledge the greeting "
    "2) Then, I should offer assistance. "
    "Hello! How can I assist you today?"
    ),
    }
    ]
    }

    # Save to file
    output_file = "data/chatbot/training_data_minimal.json"
    with open(output_file "w") as f: json.dump(training_datafindent = 2)
    print(f"Created minimal training data file: {output_file}")if __name__ == "__main__": create_minimal_training_data, ()
