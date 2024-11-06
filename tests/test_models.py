from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

from src.config.config import ModelConfig
import torch
from src.models.enhanced_transformer import EnhancedTransformer
import unittest

'Test module for enhanced transformer models.'
