import sys
import os
import numpy as np
import cv2
import scipy.io
import csv
import copy
import core.model
import os
import torch.utils.data
from core import model
from LFW_loader import LFW
from config import LFW_DATA_DIR
import argparse

def parseList(root, csv_file='pairs.csv', folder_name='lfw-deepfunneled/lfw-deepfunneled'):

    csv_path = os.path.join(root, csv_file)  
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")

    nameLs = []
    nameRs = []
    folds = []
    flags = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # ← Skip The header line
        for i, row in enumerate(reader):
            row = [x.strip() for x in row if x.strip()]  # Clean empty cells
            if len(row) == 3:       # Same person: name, img1, img2
                name, idx1, idx2 = row
                p1 = f"{name}/{name}_{int(idx1):04d}.jpg"
                p2 = f"{name}/{name}_{int(idx2):04d}.jpg"
                flag = 1
            elif len(row) == 4:     # Different people: name1, idx1, name2, idx2
                name1, idx1, name2, idx2 = row
                p1 = f"{name1}/{name1}_{int(idx1):04d}.jpg"
                p2 = f"{name2}/{name2}_{int(idx2):04d}.jpg"
                flag = -1
            else:
                print(f"Skipping invalid row {i+1}: {row}")
                continue

            # Full paths
            nameL = os.path.join(root, folder_name, p1)
            nameR = os.path.join(root, folder_name, p2)
            nameLs.append(nameL)
            nameRs.append(nameR)
            folds.append(i // 600)   # 10-fold: 600 pairs per fold
            flags.append(flag)

    print(f"Loaded {len(nameLs)} LFW pairs (10-fold) from {csv_file}")
    return [nameLs, nameRs, folds, flags]

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']
        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)
        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
    return ACCs


def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model.MobileFacenet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    nl, nr, flods, flags = parseList(lfw_dir)
    lfw_dataset = LFW(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32,
                                              shuffle=False, num_workers=8, drop_last=False)
    featureLs = None
    featureRs = None
    count = 0

    for data in lfw_loader:
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        count += data[0].size(0)
        print('extracing deep features from the face pair {}...'.format(count))
        res = [net(d).data.cpu().numpy()for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # featureLs.append(featureL)
        # featureRs.append(featureR)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--lfw_dir', type=str, default=LFW_DATA_DIR, help='The path of lfw data')
    parser.add_argument('--resume', type=str, default='./model/best/068.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_dir', type=str, default='./result/best_result.mat',
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()

    getFeatureFromTorch(args.lfw_dir, args.feature_save_dir, args.resume)
    ACCs = evaluation_10_fold(args.feature_save_dir)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
