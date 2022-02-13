from dataloader import VideoDataSet
from torch.utils.data import DataLoader
from torchvision import transforms
from CRNN import CRNN
from train import train_loop
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch import nn


device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = CRNN().to(device=device)
epochs = 200 
learning_rate = 1e-4
optimizer = Adam(model.parameters(), lr=learning_rate)
# Cross Entropy Loss
criterion = CrossEntropyLoss()
transformers = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
ds = VideoDataSet("data/videos.csv", transformers, 15)
dl = DataLoader(ds, shuffle=True)

train_loop(dl, model, optimizer, criterion, epochs, device)
