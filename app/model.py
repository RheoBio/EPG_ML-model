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


class EPGTransformerCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EPGTransformerCNN, self).__init__()
        #self.cnn = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), #[16,size/2]
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 3, 1, 1),  #[32, size/4]
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.cnn2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1), #[8,16,size/2]
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, 1, 1),  #[16,8,size/4]
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*input_size, 1000),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(1000, 150),
            nn.Dropout(0.15),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(16*8*input_size, 5000),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(1000, 150),
            nn.Dropout(0.15),
        )
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=2)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_cwt):
        x = x.unsqueeze(1)  # (batch_size, 1, size)
        out_x = self.cnn1d(x)
        out_x = out_x.view(out_x.size()[0], -1)  # (batch_size, 16*size)
        out_x = self.fc1(out_x)

        x_cwt = x_cwt.unsqueeze(1)
        out_cwt = self.cnn2d(x_cwt)
        out_cwt = out_cwt.view(out_cwt.size()[0], -1)
        out_cwt = self.fc2(out_cwt)

        out = torch.cat((out_x, out_cwt), dim=1)

        out = self.transformer(out, out)
        out = self.out(out)
        out = self.sigmoid(out)
        return out