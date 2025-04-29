To run the CNN models:
Run the cnn_runner.py file.
It is currently set to train a HCNN using channel five with weighting.
Hyperparameters are at the top of the file. Change to model='RESNET50' to train the RESNET model and model='HCNN' to train the HCNN model.
Specify the channels you want to train on by setting the BAND channel to the list of channels to use.
EXT specifies filename extensions for saving files.
You can load models by setting the LOAD_MODEL param to true and using the path to your desired model.
Batch size, lr, weight decay parameters are as listed in the params.
