from dataloader import VideoDataSet
from torch import nn
from torchvision import transforms
from torch.utils.data import DataSet, DataLoader

transformers = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std[0.229, 0.224, 0.225]),
    ])
ds = VideoDataSet("data/videos.csv", transformers)
dl = DataLoader(ds, batch_size=4,num_workers=2, shuffle=True)
    
classes = ('pass', 'shot', 'save')



