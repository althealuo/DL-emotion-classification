# Comparative Study of LSTM, CNN, and SNN Models for EEG-Based Emotion Classification

## LSTM
`main.py` contains the code to train the LSTM model on the SEED dataset, save the trained model, and plot loss and accuracy curves for the training.

Parameters such as the number of LSTM layers and the learning rate can be modified at the top of the file.

## CNN
To run the CNN models:
Run the `cnn_runner.py` file.
It is currently set to train an HCNN using channel five with weighting.

Hyperparameters are at the top of the file. Change to `model='RESNET50'` to train the RESNET model and `model='HCNN'` to train the HCNN model.
Specify the channels you want to train on by setting the `BAND` channel to the list of channels to use.
`EXT` specifies filename extensions for saving files.
You can load models by setting the `LOAD_MODEL` param to true and using the path to your desired model.
Batch size, lr, weight decay parameters are as listed in the params.

## SNN
`snn.ipynb` contains the code to train the SNN model on the SEED dataset, save the trained model, and plot loss and accuracy curves for the training.
