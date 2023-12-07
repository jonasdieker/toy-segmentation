import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class CustomDataset(Dataset):
    def __init__(self, data_root: str):
        super().__init__()

        self.image_paths = sorted(glob(f"{data_root}/images/*.png"))
        self.gt_paths = sorted(glob(f"{data_root}/masks/*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # load data
        image = Image.open(self.image_paths[index]).convert("RGB")
        gt = Image.open(self.gt_paths[index]).convert("RGB")

        # transform data
        transforms = T.Compose(
            [
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.3),
                T.ConvertImageDtype(torch.float),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.ToTensor(),
            ]
        )
        image = transforms(image)
        return (image, gt)
