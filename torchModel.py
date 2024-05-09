import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Define constants
dataset_path = '/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/SpeckleRobotDataset_Upgraded'
img_height, img_width = 224, 224
num_classes = 14
batch_size = 64
num_epochs = 10
save_path = 'TrainedModels\model_aug.pth'

use_resnet_50 = True
use_resnet_18 = False
use_moblilenet_v3 = False
use_sqeezenet = False

# Define data preprocessing and augmentation transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        #transforms.RandomResizedCrop(img_height),
        #transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        #transforms.CenterCrop(img_height),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Load images and labels
file_paths = []
labels = []
for folder_name in tqdm(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if not file_path.endswith('.txt'):
                file_paths.append(file_path)
                labels.append(folder_name)

# Encode labels
print("Finished loading images and labels.")
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Split dataset
file_paths_train, file_paths_val, labels_train, labels_val = train_test_split(
    file_paths, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Create custom datasets and data loaders
train_dataset = CustomDataset(file_paths_train, labels_train, transform=data_transforms['train'])
val_dataset = CustomDataset(file_paths_val, labels_val, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet model
if use_resnet_50:
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif use_resnet_18:
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif use_moblilenet_v3:
    model = models.mobilenet_v3_large(pretrained=True)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
elif use_sqeezenet:
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes

# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False

if use_resnet_50 or use_resnet_18:
    for param in model.fc.parameters():
        param.requires_grad = True
elif use_moblilenet_v3:
    for param in model.classifier[-1].parameters():
        param.requires_grad = True
elif use_sqeezenet:
    for param in model.classifier.parameters():
        param.requires_grad = True

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)  # Optimize all parameters

# Move model to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Training loop

train_losses = []
train_accuracies = []
total_t = 0
correct_t = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted_t = torch.max(outputs, 1)
        total_t += labels.size(0)
        correct_t += (predicted_t == labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_t / total_t
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    val_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Training Loss: {epoch_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}')

# Compute confusion matrix
conf_matrix = confusion_matrix(val_targets, val_predictions)
print("Confusion Matrix:")
print(conf_matrix)

print('Training finished.')



torch.save(model, save_path)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print('Training finished.')