# %%
import torch
import unittest

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import the module
from unet.u_net import UNet
from unet.u_net import crop_to_target_size  # Replace with the actual module you are using

class TestUNet(unittest.TestCase):

    def setUp(self):
        self.model = UNet()

    def test_forward_pass(self):
        # Test the forward pass of the model
        input_tensor = torch.rand((1, 1, 572, 572))
        output = self.model.forward(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, 2, 388, 388]))  # Adjust the expected shape based on your model

    def test_crop_to_target_size(self):
        # Test the crop_to_target_size function
        tensor = torch.rand((1, 1, 572, 572))
        target_tensor = torch.rand((1, 1, 388, 388))  # Adjust the size based on your model
        cropped_tensor = crop_to_target_size(tensor, target_tensor)
        self.assertEqual(cropped_tensor.shape, target_tensor.shape)

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)    # Notebook things, ArgumentError: argument -f/--failfast: ignored explicit argument 
# %%
