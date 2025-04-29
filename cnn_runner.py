from hcnn_model import HCNN
from cnn_model import ModifiedResnet50, CustomCNN
from utils import SeedDatasetCNN
from main import plot_training_curves

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Subset
import csv
import pickle
import warnings

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

SEED = 33

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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

MAPPING = torch.from_numpy(np.concatenate((
    np.arange(3, 6),
    np.array([2]),
    np.array([6]),
    np.arange(9, 54),
    np.arange(55, 62),
    np.arange(65, 70)
)))

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GRADIENT_CLIPPING_MAX_NORM = 0.5
MAX_EPOCHS = 5000
EARLY_STOPPING_PATIENCE = 10000
PRINT_FREQUENCY_EPOCHS = 100
# BAND = [x for x in range(5)]
BAND = [4]
WEIGHT_DECAY = 5e-3

EXT = '5_5_weighted'
MODEL = 'HCNN'
SAVE_CLASSIFIER_AS = f'{MODEL}_{EXT}.pt'

LOAD_MODEL = False
MODEL_TO_LOAD = ''

NUM_EXAMPLES = np.array([5976, 7290, 11106, 9864, 9972, 8318, 10242])
LABEL_WEIGHTS = NUM_EXAMPLES.min() / NUM_EXAMPLES

print(f'Label weights: {LABEL_WEIGHTS}')

def main():
    (train_losses, train_accuracies, validation_losses,
     validation_accuracies) = train_HCNN()
    plot_training_curves(
                        train_losses, 
                        train_accuracies, 
                        validation_losses,
                        validation_accuracies,
                        loss_filename=f'loss_{MODEL}_{EXT}.png',
                        accuracy_filename=f'accuracy_{MODEL}_{EXT}.png'
                        )

def get_CNN_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SeedDatasetCNN(channel=BAND)
    row_count = len(dataset)
    # inds = list(range(row_count))
    # random.shuffle(inds)

    # train_indices = inds[:int(TRAIN_SPLIT * row_count)]
    # validation_indices = inds[int(TRAIN_SPLIT * row_count):int((TRAIN_SPLIT + VALIDATION_SPLIT) * row_count)]
    # test_indices = inds[int((TRAIN_SPLIT + VALIDATION_SPLIT) * row_count):]

    validation_indices = list(range(80))
    test_indices = list(range(80, 160))
    train_indices = list(range(160, row_count))
    with open(f'index_{MODEL}_{EXT}.pkl', 'wb') as f:
        pickle.dump(
            {
            'train': train_indices,
            'test': test_indices,
            'validation': validation_indices
            },
            f
        )
    # train_indices = list(range(0, int(TRAIN_SPLIT * row_count)))
    # validation_indices = list(range(
    #     int(TRAIN_SPLIT * row_count),
    #     int((TRAIN_SPLIT + VALIDATION_SPLIT) * row_count)))
    # test_indices = list(range(
    #     int((TRAIN_SPLIT + VALIDATION_SPLIT) * row_count), row_count))
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    def collate_sequence(batch: list[tuple[torch.Tensor, int]]) -> tuple:
        sequences = [sequences_and_label[0]
                    for sequences_and_label in batch]
        labels = [sequences_and_label[1]
                    for sequences_and_label in batch]
        # Shape: (batch size, max sequence length, input size)
        # padded_sequences = pad_sequence(sequences, batch_first=True)
        return sequences, labels
    
    def collate_continuous(batch: list[tuple[np.ndarray, int]]) -> tuple:
        sequences = []
        labels = []
        for pair in batch:
            sequence = pair[0]
            label = pair[1]
            sequences.append(sequence)
            continuous_label = torch.repeat_interleave(label, sequence.shape[0])
            labels.append(continuous_label)
        sequences = torch.concat(sequences, 0)
        labels = torch.concat(labels, 0)

        # print(np.unique(labels.cpu().numpy(), return_counts=True))

        return sequences, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_continuous)
    # validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
    #                                shuffle=True, collate_fn=collate_sequence)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False,  collate_fn=collate_continuous)
    # validation_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                                shuffle=True, collate_fn=collate_sequence)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_sequence)
    return train_loader, validation_loader, test_loader

def train_epoch_CNN(model: nn.Module, data_loader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0
    total_count = 0
    correct_count = 0
    unique_counts = np.zeros(7)
    for batch in data_loader:
        optimizer.zero_grad()
        sequences, labels = batch
        # unique_labels, counts = np.unique(labels.cpu().numpy(), return_counts=True)
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),
        #                                max_norm=GRADIENT_CLIPPING_MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        _, predictions = torch.max(outputs, dim=1)
        total_count += labels.shape[0]
        correct_count += (predictions == labels).sum().item()
        # unique_counts[unique_labels] += counts
    # print(unique_counts)
    loss = total_loss / total_count
    accuracy = correct_count / total_count
    return loss, accuracy

def evaluate_CNN(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
             device: torch.device, continuous=False,) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    total_count = 0
    correct_count = 0
    cm = np.zeros((len(EMOTIONS), len(EMOTIONS)))
    all_pred = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            sequences, labels = batch

            if not continuous:
                # For each video
                chunk_sizes = [x.shape[0] for x in sequences]
                sequences = torch.concat(sequences, 0).to(device)
                labels = torch.stack(labels, 0).to(device)
                outputs = model(sequences)
                
                # Split into each video
                outputs_weighted = torch.zeros(len(batch[1]), len(EMOTIONS))
                clips = torch.split(outputs, chunk_sizes)
                for i, clip in enumerate(clips):
                    outputs_weighted[i, :] = clip.mean(0)
                    # guesses = clip.argmax(1)
                    # guess = guesses.mode().values
                outputs_weighted = outputs_weighted.to(device)
                loss = criterion(outputs_weighted, labels)
                total_loss += loss.item() * labels.shape[0]
                _, predictions = torch.max(outputs_weighted, dim=1)

            else:
                # For continuous
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.shape[0]
                _, predictions = torch.max(outputs, dim=1)

            total_count += labels.shape[0]
            correct_count += (predictions == labels).sum().item()
            all_pred.append(predictions.cpu())
            all_labels.append(labels.cpu())

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cm += confusion_matrix(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), labels=list(EMOTIONS.values()))

    loss = total_loss / total_count
    accuracy = correct_count / total_count
    pred = torch.cat(all_pred).flatten()
    labels = torch.cat(all_labels).flatten()
    f1 = f1_score(labels, pred, labels=list(EMOTIONS.values()), average='micro')

    return loss, accuracy, cm, f1

def train_HCNN():
    train_loader, validation_loader, test_loader = get_CNN_data_loaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}.')
    if MODEL == 'HCNN':
        model = HCNN(input_channels=len(BAND)).to(device)
    elif MODEL == 'RESNET50':
        model = ModifiedResnet50(num_channels=len(BAND)).to(device)
    elif MODEL == 'CustomCNN':
        model = CustomCNN(input_channels=len(BAND)).to(device)

    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_TO_LOAD))
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(LABEL_WEIGHTS).float().to(device))
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_validation_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0
    for epoch in range(MAX_EPOCHS):
        train_loss, train_accuracy = train_epoch_CNN(
            model, train_loader, optimizer, criterion, device)
        validation_loss, validation_accuracy, _, _ = evaluate_CNN(
            model, validation_loader, criterion, device, continuous=True)
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
        # if False:
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
    test_loss, test_accuracy, cm, f1 = evaluate_CNN(model, test_loader, criterion, device)
    cm_disp = ConfusionMatrixDisplay(cm)
    cm_disp.plot().figure_.savefig(f'confusion_matrix_{MODEL}_{EXT}.png')
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Test F1: {f1:.4f}')
    torch.save(model.state_dict(), SAVE_CLASSIFIER_AS)
    print(f'Model saved as {SAVE_CLASSIFIER_AS}.')

    run_info = {
        'Model': MODEL,
        'Bands': BAND,
        'Train %': TRAIN_SPLIT,
        'Val %': VALIDATION_SPLIT,
        'Test %': 1-TRAIN_SPLIT-VALIDATION_SPLIT,
        'lr': LEARNING_RATE,
        'Weight decay': WEIGHT_DECAY,
        'Max epochs': MAX_EPOCHS,
        'Patience': EARLY_STOPPING_PATIENCE,
        'Epochs ran': epoch,
        'Train acc': train_accuracy,
        'Train loss': train_loss,
        'Val acc': validation_accuracy,
        'Val loss':  validation_loss,
        'Test acc': test_accuracy,
        'Test loss': test_loss,
        'Test F1 Score': f1,
    }

    with open(f'run_info_{MODEL}_{EXT}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = run_info.keys())
        writer.writeheader()
        writer.writerow(run_info)

    return (train_losses, train_accuracies, validation_losses,
            validation_accuracies)

if __name__=='__main__':
    main()