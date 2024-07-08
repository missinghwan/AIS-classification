
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        #self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        #self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(1600, 32)
        #self.fc1 = nn.Linear(32 * 28 * 28, 256)
        #self.fc1 = nn.Linear(16 * 26 * 26, 256)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(2304, 32)

    def forward(self, x):
        # CNN-LSC
        #x = self.pool1(self.relu1(self.conv1(x)))
        #x = self.pool2(self.relu2(self.conv2(x)))
        #x = self.pool3(self.relu3(self.conv3(x)))
        #x = self.pool4(self.relu4(self.conv4(x)))
        #x = x.view(x.size(0), -1)  # Flatten the tensor
        #x = self.dropout(self.relu5(self.fc1(x)))
        #x = self.fc2(x)

        # CNN-SMMC
        x = self.relu1(self.conv1(x))
        x = self.pool1(self.relu1(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu1(self.conv3(x))
        x = self.pool1(self.relu1(self.conv4(x)))
        x = self.dropout(x)
        x = self.relu1(self.conv5(x))
        x = self.pool1(self.relu1(self.conv6(x)))
        x = self.dropout(x)
        x = self.relu1(self.conv7(x))
        x = self.pool1(self.relu1(self.conv8(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x
