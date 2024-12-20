import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.functional import dice
from tqdm import tqdm
import os

from dataset import CustomDataset
from segformer import SegFormer
from unet import UNet
from visualize import convert_class_idx_2_rgb

device = "cuda" if torch.cuda.is_available() else "cpu"
jaccard = JaccardIndex(task="multiclass", num_classes=12).to(device)


def plot_data(image, gt):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.moveaxis(image.numpy(), 0, -1))
    axs[1].imshow(gt)
    plt.show()


def train_val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: CrossEntropyLoss,
    epoch: int,
    device: str,
    optimizer: Optimizer = None,
    train: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0
    dice_score, iou = [], []
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        pbar.set_description(
            f"[{'Train' if train else 'Val'} epoch {str(epoch+1).zfill(3)}]"
        )
        images, gts = data

        # plot_data(images[0], convert_class_idx_2_rgb(gts[0]))

        images = images.to(device).double()
        gts = gts.to(device)

        # zero gradients
        if train:
            optimizer.zero_grad()

        # pass data through model
        preds = model(images)
        loss = loss_fn(preds, gts)

        # compute gradients and adjust weights
        if train:
            loss.backward()
            optimizer.step()

        # update progress bar
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5
            pbar.set_postfix({"loss": last_loss})
            running_loss = 0
        dice_score.append(dice(preds, gts))
        iou.append(jaccard(preds, gts))
    print(f"Dice Score: {sum(dice_score)/len(dice_score)}")
    print(f"mIoU: {sum(iou)/len(iou)}")


def main():
    batch_size = 8
    max_epochs = 20
    lr = 0.0001
    betas = (0.9, 0.999)
    num_classes = 12
    data_root = "/home/jonas/Downloads/CamVid/"

    loss = CrossEntropyLoss()
    model = UNet(3, num_classes).double().to(device)
    # model = SegFormer(num_classes).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    train_dataset = CustomDataset(
        image_root=os.path.join(data_root, "train"),
        mask_root=os.path.join(data_root, "train_labels"),
    )
    val_dataset = CustomDataset(
        image_root=os.path.join(data_root, "val"),
        mask_root=os.path.join(data_root, "val_labels"),
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    for epoch in range(max_epochs):
        train_val_epoch(
            model, train_dataloader, loss, epoch, device, optimizer, train=True
        )
        train_val_epoch(model, val_dataloader, loss, epoch, device, train=False)
        torch.save(model.state_dict(), "segformer.pth")


if __name__ == "__main__":
    main()
