import torch
import torch.nn as nn
from model import CNN
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.one_hot = torch.tensor(
            [[1,0,0,0,0,0,0],
             [0,1,0,0,0,0,0],
             [0,0,1,0,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,0,1,0,0],
             [0,0,0,0,0,1,0],
             [0,0,0,0,0,0,1]],
            dtype=torch.float64
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        label = self.data.iloc[idx, 1]
        # label = self.one_hot[label-1]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transforms to be applied to the images
transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the custom dataset
# dataset = CustomDataset('train_loitering.csv', transform=transform)
# dataset = CustomDataset('train_sailing.csv', transform=transform)
dataset = CustomDataset('train_movement.csv', transform=transform)


train_size = int(len(dataset)*0.7)
# Create a data loader for the dataset
train_dataset, validation_dataset = random_split(dataset, [train_size, len(dataset)-train_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)


# Instantiate the CNN model and move it to the device
model = CNN().to(device)
#model = models.resnet18()
#model = models.resnet50()
#model = models.resnet101()
#model = models.resnet152()
#model = models.vit_b_16()
#model = models.vit_b_32()
#model = models.densenet121()
#model = models.densenet201()

# for resnet
#model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# for densenet
#model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# for ViT 16
#model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
# for ViT 32
#model.conv_proj = nn.Conv2d(1, 768, kernel_size=(32, 32), stride=(32, 32))


#model.image_size = 120

#model.fc = nn.Linear(512, 32, bias=False)
model.to(device)


# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
for epoch in range(100):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    model.train()

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels.long())

        loss.backward()
        optimizer.step()

        # acc = outputs.data.max(1)[1].eq(labels.data).sum() / len(images) * 100

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        # predicted = F.one_hot(predicted, num_classes=4)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(dataset)
    train_accuracy = train_correct / train_total

    print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Accuracy: {train_accuracy:.6f}')

print("-----------------------------")
with torch.no_grad():
    model.eval()
    val_total = 0
    val_correct = 0
    for images, labels in validation_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {(val_correct/val_total):.6f}')
