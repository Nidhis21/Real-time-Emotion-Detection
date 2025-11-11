import torch
import torch.nn as nn
import torch.nn.functional as F
class EmotionCNN(nn.Module):
    def __init__(self):
      super(EmotionCNN, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
      self.dropout25 = nn.Dropout(0.25)
      self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
      self.dropout50 = nn.Dropout(0.5)
      
      # create dummy tensor with expected input size
      dummy = torch.zeros(1, 1, 48, 48)
      dummy = self.pool(F.relu(self.conv1(dummy)))
      dummy = self.pool(F.relu(self.conv2(dummy)))
      dummy = F.relu(self.conv3(dummy))
      flattened_size = dummy.numel()  # total elements in feature tensor
      
      self.fc1 = nn.Linear(flattened_size, 128)
      self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = F.relu(self.conv2(x))
      x = self.dropout25(x)
      x = self.pool(x)
      x = F.relu(self.conv3(x))
      x = self.dropout25(x)
      
      
      
      x = torch.flatten(x, 1)
      
      
      x = F.relu(self.fc1(x))
      x = self.dropout50(x)
      x = self.fc2(x)
      return x
