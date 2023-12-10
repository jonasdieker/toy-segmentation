# Toy Segmentation

This repo serves as a refresher on segmentation using neural networks.

On top of the classic U-Net, I also added the SegFormer network. Both models are not the completely standard model, but were slightly adjusted. Mainly the SegFormer model was adapted to not rely on any other deep learning libraries other than PyTorch.

I was suprised to see that SegFormer trains significantly faster than U-Net.

Without doing too much hyper-parameter tuning these are the results obtained on the CamVid dataset:

[add table here]

