import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from scipy.io import loadmat
import pandas as pd

import numpy as np

def main():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # print(model)
    d = loadmat('../dataset/EEG_features/1.mat')
    de = loadmat('../dataset/EYE_features/1.mat')
    corder = np.genfromtxt('../dataset/Channel Order.csv', dtype=str, delimiter=',').flatten()

    mapping = np.concatenate((
        np.arange(3, 6),
        np.array([2]),
        np.array([6]),
        np.arange(9, 54),
        np.arange(55, 62),
        np.arange(65, 70)
    ))

    array = np.zeros((8 * 9), dtype=object)
    array[mapping] = corder
    array = array.reshape(8, 9)
    
    sparse = np.zeros((16, 18), dtype=object)
    sparse[::2, ::2] = array
    sparse = np.pad(sparse, ((2, 1), (1, 0)))
    print(array)
    print(sparse)
    print(sparse.shape)

    return


if __name__ == '__main__':
    main()