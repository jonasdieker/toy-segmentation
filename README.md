# Segmentation Benchmark

This repo serves as an experiments to compare CNN-based segmentation networks to the transformer-based SegFormer model.

A goal was also to make getting started with SegFormer eaiser, by not using other repo dependencies such as `mmcv` or the like.

## Results

Without doing too much hyper-parameter tuning these are the results obtained on the CamVid dataset:

|Model|Dice Score|mIoU|
|---|---|---|
|U-Net| - | - |
|DeepLabV3| - | - |
|SegFormer (B0)| - | - |

## Getting Started

 - Download the data from [here](https://www.kaggle.com/datasets/carlolepelaars/camvid)
 - Create environment and install requirements:

    ```
    conda create -n segmentation python=3.9
    conda activate segmentation
    pip install -r requirements.txt
    ```

 - Run training script `train.py`