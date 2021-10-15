import cv2 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)

    v_cap.release()
    return frames, v_len

class VideoDataSet(Dataset):
    def __init__(self, all_video_file, transformers):
        # This maps csv which has file path and label to numpy arrray 
        self.videos = np.genfromtxt(all_video_file, delimeters=",", dtype=np.unicode_)  
        self.transformers = transformers 

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):



