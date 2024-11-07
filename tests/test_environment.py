import numpy as np
import torch
import unittest

class TestEnvironment:
    pass
    def test_test_cuda_availability(self):

    if torch.cuda.is_available():
    device = torch.device("cuda")
    else:
    device = torch.device("cpu")
    self.assertIsNotNone(device)
    if torch.cuda.is_available():
    self.assertTrue(torch.cuda.is_initialized())

if __name__ == "__main__":

if __name__ == "__main__":
    unittest.main()




if __name__ == "__main__":

if __name__ == "__main__":
    unittest.main()
