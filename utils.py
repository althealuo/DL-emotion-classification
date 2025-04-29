from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

DATASET_PATH = Path('dataset')
EMOTIONS = {
    'Neutral': 0,
    'Happy': 1,
    'Sad': 2,
    'Anger': 3,
    'Fear': 4,
    'Disgust': 5,
    'Surprise': 6
}
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1

MAPPING = torch.from_numpy(np.concatenate((
    np.arange(3, 6),
    np.array([2]),
    np.array([6]),
    np.arange(9, 54),
    np.arange(55, 62),
    np.arange(65, 70)
)))

class SeedDatasetCNN(Dataset):
    def __init__(self, channel=list(range(5))):
        if not isinstance(channel, list):
            channel = [channel]

        # self.features = None
        # self.labels = None
        self.features = []
        self.labels = []
        labels = pd.read_excel(
            DATASET_PATH / 'emotion_label_and_stimuli_order.xlsx', header=None,
            usecols='B:U', skiprows=lambda row_index: row_index % 2 == 0
        )
        labels = labels.values.flatten().tolist()
        labels = [EMOTIONS[label] for label in labels]
        labels = labels * 20
        for subject_index in range(1, 21):
            subject_features = sio.loadmat(str(DATASET_PATH / 'EEG_features'
                                               / f'{subject_index}.mat'))
            for video_index in range(1, 81):
                de_features_o = subject_features[f'de_LDS_{video_index}']
                # (sequence_len, bands, channels) -> (sequence_len, bands, 19, 19)
                de_mapped = np.zeros((
                    de_features_o.shape[0],
                    len(channel),
                    8*9,
                    ),
                    dtype=np.float32
                )
                de_mapped[:, :, MAPPING] = de_features_o[:, channel, ...]
                de_sparse = np.zeros((
                    de_features_o.shape[0],
                    len(channel),
                    16,
                    18
                ))
                # de_mapped = (de_mapped - de_mapped.min((0, 1))) / (de_mapped - de_mapped.min((0, 1))).max((0, 1))
                # de_mapped = de_mapped / de_mapped.max((0, 1))
                # de_mapped = np.nan_to_num(de_mapped)
                de_sparse[..., ::2, ::2] = de_mapped.reshape((
                    de_features_o.shape[0],
                    len(channel),
                    8,
                    9
                ))
                # de_features = nn.functional.pad(
                #     torch.from_numpy(de_sparse),
                #     (0, 1, 1, 2, 0, 0, 0, 0),
                #     # (0, 0, 0, 0, 2, 1, 1, 0)
                # ).float()
                de_features = nn.functional.interpolate(
                    torch.tensor(de_mapped).reshape((
                        de_features_o.shape[0],
                        len(channel),
                        8,
                        9
                    )),
                    size=(20, 20),
                    mode='bilinear'
                ).float()
                # de_features = (de_features - de_features.mean()) / (de_features.std() + 1e-6)
                # de_features = de_features.reshape(de_features.shape[0], -1)
                # Shape: (sequence_len, bands, 19, 19)

                # if self.features is None:
                #     self.features = de_features
                #     self.labels = torch.repeat_interleave(
                #         torch.tensor([labels[video_index*subject_index-1]]),
                #         de_features.shape[0]
                #         )
                # else:
                #     self.features = torch.concat((self.features, de_features))
                #     self.labels = torch.concat((
                #         self.labels,
                #         torch.repeat_interleave(
                #             torch.tensor([labels[video_index*subject_index-1]]),
                #             de_features.shape[0]
                #         )
                #     ))
                self.features.append(de_features)
                # self.features.append(torch.tensor(de_mapped.reshape((-1, len(channel), 8, 9))))
                # self.labels.append(torch.tensor(labels[video_index*subject_index-1]))
                self.labels.append(torch.tensor(labels[(video_index - 1) + ((subject_index - 1) * 80)]))
        print(np.unique(np.array(labels), return_counts=True))
                

    def __len__(self) -> int:
        # return self.labels.shape[0]
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[np.ndarray, int]:
        return self.features[idx], self.labels[idx]
    


class SeedDatasetNN(Dataset):
    def __init__(self, channel=list(range(5))):
        if not isinstance(channel, list):
            channel = [channel]

        # self.features = None
        # self.labels = None
        self.features = []
        self.labels = []
        labels = pd.read_excel(
            DATASET_PATH / 'emotion_label_and_stimuli_order.xlsx', header=None,
            usecols='B:U', skiprows=lambda row_index: row_index % 2 == 0
        )
        labels = labels.values.flatten().tolist()
        labels = [EMOTIONS[label] for label in labels]
        labels = labels * 20
        for subject_index in range(1, 21):
            subject_features = sio.loadmat(str(DATASET_PATH / 'EEG_features'
                                               / f'{subject_index}.mat'))
            for video_index in range(1, 81):
                de_features_o = subject_features[f'de_LDS_{video_index}']
                # (sequence_len, bands, channels) -> (sequence_len, bands, 19, 19)
                de_features = de_features_o[:, channel, ...]
                self.features.append(de_features)
                # self.features.append(torch.tensor(de_mapped.reshape((-1, len(channel), 8, 9))))
                # self.labels.append(torch.tensor(labels[video_index*subject_index-1]))
                self.labels.append(torch.tensor(labels[(video_index - 1) + ((subject_index - 1) * 80)]))
        print(np.unique(np.array(labels), return_counts=True))
                

    def __len__(self) -> int:
        # return self.labels.shape[0]
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[np.ndarray, int]:
        return self.features[idx], self.labels[idx]