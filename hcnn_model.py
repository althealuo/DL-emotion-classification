import torch
from torch import nn

class HCNN(nn.Module):
    def __init__(self, input_channels=5, output_size=7):
        super(HCNN, self).__init__()
        # Assume input feature maps are 19x19
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, output_size),
            # nn.Softmax()
        )

    def forward(self, batch) -> torch.Tensor:
        features = self.feature_extractor(batch)
        outputs = self.classifier(features)
        return outputs