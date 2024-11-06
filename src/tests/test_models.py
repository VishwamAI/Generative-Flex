from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch



from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch



from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os



import torch.nn as nn



from src.config.config import ModelConfig
from src.models.transformer import TransformerModel
import unittest




class TestModels:
    """
    Class implementing TestModels functionality.
    """

    Module for implementing specific functionality.
    Method for setUp..
    
    Method for test_transformer_model..
    
    Method for test_model_forward..
    """
    model = TransformerModel(self.config)
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    outputs = model(input_ids)
    self.assertEqual(outputs.shape, (batch_size, seq_length, self.config.hidden_size))

    if __name__ == '__main__':
    unittest.main()
