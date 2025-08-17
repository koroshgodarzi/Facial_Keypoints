
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
# from supervision import Detections
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_load import Rescale, RandomCrop
import matplotlib.image as mpimg
import pandas as pd
from torchvision.models import densenet121


class CustomDenseNet(torch.nn.Module):
    def __init__(self):
        super(CustomDenseNet, self).__init__()
        self.base_model = densenet121(pretrained=True)
        self.fc = torch.nn.Linear(50176, 136)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.base_model.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)

        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class Normalize_rgb():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image = image.astype(np.float32) / 255.0

        image[:, :, 0] = (image[:, :, 0] - self.mean[0]) / self.std[0]
        image[:, :, 1] = (image[:, :, 1] - self.mean[1]) / self.std[1]
        image[:, :, 2] = (image[:, :, 2] - self.mean[2]) / self.std[2]

        key_pts = (key_pts - 100)/50.0

        return {'image': image, 'keypoints': key_pts}


def YOLO_pose(number_of_epochs):
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    model = YOLO('custom_YOLO_pose.yaml')
    model = YOLO(model_path)
    model = YOLO('custom_YOLO_pose.yaml').load('yolov8n-pose.pt')

    backbone = model.model.model[: -1]
    for param in backbone.parameters():
        param.requires_grad = False

    train_results = model.train(data="data/data.yaml", epochs=number_of_epochs, imgsz=224)

    return model, train_results


if __name__ == '__main__':
    pass
