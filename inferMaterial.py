import speckleRobotServer
import torch
from torchvision import transforms
from PIL import Image
import io


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Start the server and receive an image
server, connection, address = speckleRobotServer.startServer()
img = speckleRobotServer.receiveShot(connection)
img = Image.open(io.BytesIO(img))

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the entire model
model = torch.load('TrainedModels/model.pth')
model.to(device)
model.eval()  # Set the model to evaluation mode

# Transform image
img = transform(img)
img = img.unsqueeze(0)  # Add a batch dimension
img = img.to(device)

print('reapering to predicy')

# Assuming the model and img are already loaded and transformed
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted class: {predicted.item()}')


