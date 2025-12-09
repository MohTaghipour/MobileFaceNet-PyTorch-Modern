This project is based on the original implementation by Xiaoccer:

https://github.com/Xiaoccer/MobileFaceNet_Pytorch

“This project is still under development, so please use it with caution.”

Currently it is based on:

- CUDA: 12.4
- PyTorch: 2.5.1
- torchvision: 0.20.1

This version includes:

- Full upgrade to PyTorch 2.x
- Modern project structure
- Debug fixes
- New training utilities

In order to train use this dataset:
https://www.kaggle.com/datasets/ntl0601/casia-webface

For test:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Change config.py according to your setup.

Run train.py. You should see something like this:

Visible GPUs: [0]
Loaded 460412 images from 10537 identities.
Loaded 6000 LFW pairs (10-fold) from pairs.csv

Starting epoch 1/25  (3596 batches total)
 Batch 1/3596 (0.0%)
 Batch 21/3596 (0.6%)
 Batch 41/3596 (1.1%)
 Batch 61/3596 (1.7%)
 Batch 81/3596 (2.3%) 
