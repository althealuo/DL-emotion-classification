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

HIDDEN_SIZE = 256
LAYER_COUNT = 4

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
GRADIENT_CLIPPING_MAX_NORM = 0.5
MAX_EPOCHS = 3000
EARLY_STOPPING_PATIENCE = 200
PRINT_FREQUENCY_EPOCHS = 10


class SeedDataset(Dataset):
    def __init__(self):
        self.features = []
        for subject_index in range(1, 21):
            subject_features = sio.loadmat(str(DATASET_PATH / 'EEG_features'
                                               / f'{subject_index}.mat'))
            for video_index in range(1, 81):
                de_features = subject_features[f'de_LDS_{video_index}']
                # Flatten the frequency band and EEG channel dimensions.
                de_features = de_features.reshape(de_features.shape[0], -1)
                # Shape: (sequence length, input size (5 * 62))
                self.features.append(de_features)
        labels = pd.read_excel(
            DATASET_PATH / 'emotion_label_and_stimuli_order.xlsx', header=None,
            usecols='B:U', skiprows=lambda row_index: row_index % 2 == 0
        )
        labels = labels.values.flatten().tolist()
        labels = [EMOTIONS[label] for label in labels]
        labels = labels * 20
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[np.ndarray, int]:
        return self.features[idx], self.labels[idx]


def get_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SeedDataset()
    row_count = len(dataset)
    train_indices = list(range(0, int(TRAIN_SPLIT * row_count)))
    validation_indices = list(range(
        int(TRAIN_SPLIT * row_count),
        int((TRAIN_SPLIT + VALIDATION_SPLIT) * row_count)))
    test_indices = list(range(
        int((TRAIN_SPLIT + VALIDATION_SPLIT) * row_count), row_count))
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    def collate_fn(batch: list[tuple[np.ndarray, int]]) -> tuple:
        sequences = [torch.tensor(sequences_and_label[0], dtype=torch.float)
                     for sequences_and_label in batch]
        labels = torch.tensor([sequences_and_label[1]
                               for sequences_and_label in batch],
                              dtype=torch.long)
        sequence_lengths = torch.tensor([sequence.shape[0]
                                         for sequence in sequences],
                                        dtype=torch.long)
        # Shape: (batch size, max sequence length, input size)
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, sequence_lengths, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)
    return train_loader, validation_loader, test_loader


class LstmClassifier(nn.Module):
    def __init__(self):
        super(LstmClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=5 * 62, hidden_size=HIDDEN_SIZE,
                            num_layers=LAYER_COUNT, batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, len(EMOTIONS))

    def forward(self, batch: torch.Tensor,
                sequence_lengths: torch.Tensor) -> torch.Tensor:
        packed_batch = pack_padded_sequence(batch, sequence_lengths.cpu(),
                                            batch_first=True,
                                            enforce_sorted=False)
        _, (final_hidden_states, _) = self.lstm(packed_batch)
        output = self.linear(final_hidden_states[-1])
        return output


def train_epoch(model: nn.Module, data_loader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0
    total_count = 0
    correct_count = 0
    for batch in data_loader:
        optimizer.zero_grad()
        sequences, sequence_lengths, labels = batch
        sequences = sequences.to(device)
        sequence_lengths = sequence_lengths.to(device)
        labels = labels.to(device)
        outputs = model(sequences, sequence_lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=GRADIENT_CLIPPING_MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        _, predictions = torch.max(outputs, dim=1)
        total_count += labels.shape[0]
        correct_count += (predictions == labels).sum().item()
    loss = total_loss / total_count
    accuracy = correct_count / total_count
    return loss, accuracy


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for batch in data_loader:
            sequences, sequence_lengths, labels = batch
            sequences = sequences.to(device)
            sequence_lengths = sequence_lengths.to(device)
            labels = labels.to(device)
            outputs = model(sequences, sequence_lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.shape[0]
            _, predictions = torch.max(outputs, dim=1)
            total_count += labels.shape[0]
            correct_count += (predictions == labels).sum().item()
    loss = total_loss / total_count
    accuracy = correct_count / total_count
    return loss, accuracy


def train() -> tuple[list[float], list[float], list[float], list[float]]:
    train_loader, validation_loader, test_loader = get_data_loaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}.')
    model = LstmClassifier().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_validation_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0
    for epoch in range(MAX_EPOCHS):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, criterion, device)
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        if epoch == 0 or (epoch + 1) % PRINT_FREQUENCY_EPOCHS == 0:
            print(f'Epoch {epoch + 1}: '
                  f'Train loss: {train_loss:.4f}, '
                  f'Train accuracy: {train_accuracy:.4f}, '
                  f'Validation loss: {validation_loss:.4f}, '
                  f'Validation accuracy: {validation_accuracy:.4f}')
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Stopping because validation loss did not improve for '
                  f'{EARLY_STOPPING_PATIENCE} epochs. Best validation loss: '
                  f'{best_validation_loss:.4f}')
            break
    model.load_state_dict(best_model)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')
    torch.save(model.state_dict(), 'lstm_classifier.pt')
    print('Model saved as `lstm_classifier.pt`.')
    return (train_losses, train_accuracies, validation_losses,
            validation_accuracies)


def plot_training_curves(train_losses: list[float],
                         train_accuracies: list[float],
                         validation_losses: list[float],
                         validation_accuracies: list[float]):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, validation_losses, label='Validation loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    print('Loss and accuracy curves saved as `loss.png` and `accuracy.png`.')


def main():
    (train_losses, train_accuracies, validation_losses,
     validation_accuracies) = train()
    plot_training_curves(train_losses, train_accuracies, validation_losses,
                         validation_accuracies)


if __name__ == '__main__':
    main()
