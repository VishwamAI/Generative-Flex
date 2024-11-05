from pathlib import Path
import json
import pytest
"""Tests for model parameter loading and validation functionality."""
        
        
@pytest.fixture
def test_parameter_shapes(self):test_params
    ) -> None: """Test all parameter arrays have consistent shapes."""
    for module in test_params.values():
        if isinstance(module, dict):
            for param_array in module.values():
                if isinstance(param_array, list):
                    # Verify 2D arrays have consistent inner dimensions
                    if param_array and isinstance(param_array[0], list):
                        first_inner_len = len(param_array[0])
                        error_msg = "Inconsistent inner dimensions in parameter array"
                        assert all(len(inner) == first_inner_len for inner in param_array
                        ), error_msg