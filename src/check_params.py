import json
import pprint

def main():
    # Load and examine parameter structure
    with open('model_params_minimal.json', 'r') as f:
        params = json.load(f)

    print("Parameter Structure:")
    print("===================")
    print("\nKeys at root level:")
    print(list(params.keys()))

    print("\nDetailed structure:")
    pprint.pprint(params, depth=4)

    print("\nShape information:")
    for key in params:
        if isinstance(params[key], dict):
            print(f"\n{key}:")
            for subkey, value in params[key].items():
                if isinstance(value, list):
                    print(f"  {subkey}: shape = ({len(value)}, {len(value[0]) if value and isinstance(value[0], list) else 1})")

if __name__ == "__main__":
    main()
