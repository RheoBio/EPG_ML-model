#Library imports
import os
import time
import random
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import pandas as pd
import numpy as np

import cv2
from sklearn.model_selection import train_test_split
import torch

class EPGCNN(nn.Module):
    def __init__(self):
        super(EPGCNN, self).__init__()
        #self.cnn = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), #[16,220]
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 3, 1, 1),  #[32, 110]
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 3, 1, 1),  #[64, 55]
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*110, 1100),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(1100, 560),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(560, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid(),
         )



    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, 440)
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # (batch_size, 16*440)
        out = self.fc(out)
        #out = out.squeeze()
        return out