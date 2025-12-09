## MobileFaceNet with Modern Pytorch

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

1. Update `config.py` according to your setup.

2. Run `train.py`. You should see output similar to the following (depending on your `batch_size` and number of GPUs):

_Visible GPUs: [0]_

_Loaded 460412 images from 10537 identities._

_Loaded 6000 LFW pairs (10-fold) from pairs.csv_

_Starting epoch 1/25  (3596 batches total)_

_Batch 1/3596 (0.0%)_

_Batch 21/3596 (0.6%)_

_Batch 41/3596 (1.1%)_

_Batch 61/3596 (1.7%)_

etc.
- The numbers may vary depending on your dataset, batch size, and GPU configuration.
- The progress updates will continue for all batches in each epoch.
