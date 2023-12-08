from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

RGBLabel2LabelName = {
    (128, 128, 128): "Sky",
    (0, 128, 64): "Building",
    (128, 0, 0): "Building",
    (64, 192, 0): "Building",
    (64, 0, 64): "Building",
    (192, 0, 128): "Building",
    (192, 192, 128): "Pole",
    (0, 0, 64): "Pole",
    (128, 64, 128): "Road",
    (128, 0, 192): "Road",
    (192, 0, 64): "Road",
    (0, 0, 192): "Sidewalk",
    (64, 192, 128): "Sidewalk",
    (128, 128, 192): "Sidewalk",
    (128, 128, 0): "Tree",
    (192, 192, 0): "Tree",
    (192, 128, 128): "SignSymbol",
    (128, 128, 64): "SignSymbol",
    (0, 64, 64): "SignSymbol",
    (64, 64, 128): "Fence",
    (64, 0, 128): "Car",
    (64, 128, 192): "Car",
    (192, 128, 192): "Car",
    (192, 64, 128): "Car",
    (128, 64, 64): "Car",
    (64, 64, 0): "Pedestrian",
    (192, 128, 64): "Pedestrian",
    (64, 0, 192): "Pedestrian",
    (64, 128, 64): "Pedestrian",
    (0, 128, 192): "Bicyclist",
    (192, 0, 192): "Bicyclist",
    (0, 0, 0): "Void",
}

LabelName2LabelIndex = {
    "Sky": 0,
    "Building": 1,
    "Pole": 2,
    "Road": 3,
    "Sidewalk": 4,
    "Tree": 5,
    "SignSymbol": 6,
    "Fence": 7,
    "Car": 8,
    "Pedestrian": 9,
    "Bicyclist": 10,
    "Void": 11,
}


class CustomDataset(Dataset):
    def __init__(self, image_root: str, mask_root):
        super().__init__()

        self.image_paths = sorted(glob(f"{image_root}/*.png"))
        self.gt_paths = sorted(glob(f"{mask_root}/*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # load data
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        gt = np.array(Image.open(self.gt_paths[index]))
        gt = gt / (0.039 * 25)

        # map rgb value to class idx
        gt_mask = np.zeros((gt.shape[0], gt.shape[1]))
        for key, val in RGBLabel2LabelName.items():
          label_idx = LabelName2LabelIndex[val]
          gt_mask[np.all(gt == key, axis=-1)] = label_idx

        # transform data
        transforms_img = T.Compose(
            [
                T.ToTensor(),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.3),
                T.ConvertImageDtype(torch.float),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transforms_gt = T.Compose(
            [
                T.ToTensor(),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.3),
            ]
        )
        image, gt_mask = transforms_img(image), transforms_gt(gt_mask)
        return image, gt_mask.squeeze().long()
