import sys
import unittest

import torch

sys.path.append('.')
from src.models.supervised.resnet_transfer import FCNResnetTransfer

class TestFCNResnetTransfer(unittest.TestCase):

    def setUp(self):

        self.in_channels = 3
        self.out_channels = 2
        self.scale_factor = 50

        self.model = FCNResnetTransfer(self.in_channels, self.out_channels, self.scale_factor)
    
    def tearDown(self) -> None:
        return super().tearDown()