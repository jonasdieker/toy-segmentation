"""
Setup data loader
Loss
train func
val func
hyper-params
"""
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from dataset import CustomDataset
from model import UNet

# perform argmax over channels of output
# apply crossentropy
