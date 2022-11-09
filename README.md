# u-net-aerial-imagery-segmentation

This repository accompanies this [Medium Article](https://medium.com/towards-data-science/semantic-segmentation-of-aerial-imagery-using-u-net-in-python-552705238514
)
https://medium.com/@andrewdaviesul/membership


The project aims to provide an implementation of a Tensorflow U-Net model for the semantic segmentation of aerial imagery.

![sample aerial image](https://images.unsplash.com/flagged/photo-1559717865-a99cac1c95d8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2071&q=80)
Photo by ZQ Lee on Unsplash

## Dataset

The MBRSC dataset exists under the CC0 license, available to download. It consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes.

Training Data Image                               |  Training Data Mask
:------------------------------------------------:|:-----------------------------------------------------:
![sample aerial image](images/image_part_001.jpg) |  ![sample aerial mask](images/image_part_001_mask.png)

## Model

A simple U-Net model is used for the semantic segmentation. The model architecture is illustrated below.
U-Net consists of two critical paths: 1) Contraction 2) Expansion

Filters in the expansive path contain high level spatial and contextual feature information.
Detailed fine-grained structural information contained in the contraction path.

![model architecture](images/unet-architecture.png)

## Prediction

Below is a visualisation of ten outputs portraying the patched aerial image of Dubai, the ground truth mask and the U-Net segmentation prediction.

![model predictions](images/predictions.png)