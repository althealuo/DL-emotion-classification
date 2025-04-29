import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
# model.fc = torch.nn.Linear(2048, 7, bias=True)
# print(model)

class ModifiedResnet50(nn.Module):

    def __init__(self, num_channels=1):
        super(ModifiedResnet50, self).__init__()

        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(2048, 7, bias=True)

    def forward(self, batch):
        return self.model(batch)
    
class CustomCNN(nn.Module):

    def __init__(self, input_channels=1):
        super(CustomCNN, self).__init__()
        # Assume input feature maps are 19x19
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 7),
            # nn.Softmax()
        )

    def forward(self, batch) -> torch.Tensor:
        features = self.feature_extractor(batch)
        outputs = self.classifier(features)
        return outputs
    
class CustomPerceptron(nn.Module):
    def __init__(self, input_size=72):
        super(CustomPerceptron, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 144),
            nn.Dropout1d(),
            nn.ReLU(),
            nn.Linear(144, 7)
        )
