import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def get_frames(filename, n_frames=3):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len - 1, v_len // n_frames, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if fn in frame_list:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            asarray = np.asarray(frame)
            frames.append(asarray)

    v_cap.release()
    # Change dimensions to Frames x Channel x Height x Width
    np_asarray = np.transpose(np.asarray(frames), (3,0, 2, 1))
    return np_asarray, len(np_asarray)


def _cut_frames(frames, length, number_of_frames_wanted):
    difference = length - number_of_frames_wanted
    half_of_frames_to_delete = difference // 2

    return frames[half_of_frames_to_delete: length - half_of_frames_to_delete]


class VideoDataSet(Dataset):
    def __init__(self, all_video_file, transformers, how_many_frames):
        # This maps csv which has file path and label to numpy arrray 
        self.videos = np.genfromtxt(all_video_file, delimiter=",", dtype=np.unicode_)
        self.transformers = transformers
        self.how_many_frames = how_many_frames
        self.video_labels = {"passes" : 0, "shots" : 1, "saves" : 2}

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        movie, label = self.videos[idx]
        frames, length = get_frames(movie)
        frames = _cut_frames(frames, length, self.how_many_frames)
        frames_torch = []

        for frame in frames:
            image = Image.fromarray(frame, "RGB")
            frame = self.transformers(image)
            frames_torch.append(frame)
        return torch.stack(frames_torch), self.video_labels.get(label)


data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
