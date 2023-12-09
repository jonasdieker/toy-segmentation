import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics.functional import dice

from dataset import CustomDataset, LabelName2LabelIndex, RGBLabel2LabelName
from model import UNet

LabelName2RGBLabel = {v: k for k, v in RGBLabel2LabelName.items()}
LabelIndex2LabelName = {v: k for k, v in LabelName2LabelIndex.items()}

def convert_class_idx_2_rgb(mask: np.ndarray) -> np.ndarray:
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int32)
    for k, v in LabelIndex2LabelName.items():
        rgb_colour = LabelName2RGBLabel[v]
        rgb_mask[mask == k] = rgb_colour
    return rgb_mask

if __name__ == "__main__":

    model = UNet(3, 12)
    loss_fn = CrossEntropyLoss()
    val_dataset = CustomDataset(
        image_root="/home/jonas/Downloads/CamVid/val", 
        mask_root="/home/jonas/Downloads/CamVid/val_labels", 
        test=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    image, gt = next(iter(val_dataloader))
    pred = model(image)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    print(pred.shape)
    loss = loss_fn(pred, gt)
    print(f"loss: {loss}, dice_score: {dice(pred, gt)}")

    # format ground truth
    gt = gt[0].numpy()
    pred = torch.argmax(pred[0], axis=0).numpy()
    rgb_gt = convert_class_idx_2_rgb(gt)
    rgb_pred = convert_class_idx_2_rgb(pred)

    # plot data
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.moveaxis(image[0].numpy(), 0, -1))
    axs[0].title.set_text("image")
    axs[1].imshow(rgb_gt)
    axs[1].title.set_text("ground truth")
    axs[2].imshow(rgb_pred)
    axs[2].title.set_text("prediction")
    plt.show()