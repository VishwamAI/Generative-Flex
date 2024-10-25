"""Tests for model parameter loading and validation functionality."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def test_params():
    """Fixture providing minimal test parameters for validation."""
    return {
        "encoder": {"weights": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.1, 0.2]},
        "decoder": {
            "attention": [[0.5, 0.5], [0.5, 0.5]],
            "mlp": [[1.0, 1.0], [1.0, 1.0]],
        },
    }


@pytest.fixture
def params_file(tmp_path, test_params):
    """Fixture creating a temporary parameter file for testing."""
    params_path = tmp_path / "model_params_minimal.json"
    with open(params_path, "w") as f:
        json.dump(test_params, f)
    return params_path


def load_params(file_path: Path) -> dict:
    """Helper function to load parameters from file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Parameter file not found: {file_path}")
    except json.JSONDecodeError:
        pytest.fail(f"Invalid JSON in parameter file: {file_path}")


def test_params_file_exists(params_file):
    """Test that parameter file exists and is readable."""
    assert params_file.exists()
    params = load_params(params_file)
    assert isinstance(params, dict)


def test_params_structure(test_params):
    """Test parameter dictionary has expected structure."""
    assert "encoder" in test_params
    assert "decoder" in test_params
    assert isinstance(test_params["encoder"], dict)
    assert isinstance(test_params["decoder"], dict)


def test_encoder_params(test_params):
    """Test encoder parameters have correct structure and shapes."""
    encoder = test_params["encoder"]
    assert "weights" in encoder
    assert "bias" in encoder
    assert isinstance(encoder["weights"], list)
    assert isinstance(encoder["bias"], list)
    assert len(encoder["weights"]) == 2
    assert len(encoder["weights"][0]) == 2
    assert len(encoder["bias"]) == 2


def test_decoder_params(test_params):
    """Test decoder parameters have correct structure and shapes."""
    decoder = test_params["decoder"]
    assert "attention" in decoder
    assert "mlp" in decoder
    assert isinstance(decoder["attention"], list)
    assert isinstance(decoder["mlp"], list)
    assert len(decoder["attention"]) == 2
    assert len(decoder["attention"][0]) == 2
    assert len(decoder["mlp"]) == 2
    assert len(decoder["mlp"][0]) == 2


def test_parameter_shapes(test_params):
    """Test all parameter arrays have consistent shapes."""
    for module in test_params.values():
        if isinstance(module, dict):
            for param_array in module.values():
                if isinstance(param_array, list):
                    # Verify 2D arrays have consistent inner dimensions
                    if param_array and isinstance(param_array[0], list):
                        first_inner_len = len(param_array[0])
                        error_msg = "Inconsistent inner dimensions in parameter array"
                        assert all(
                            len(inner) == first_inner_len for inner in param_array
                        ), error_msg
