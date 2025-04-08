# mnist_nn_human_style.py
# Eros Marcello | First pass at a simple NN for MNIST digits
# Goal: Train, save, and reload a basic model for handwritten digit recognition

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Let's use GPU if it's there, otherwise stick with CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Just keeping the dataset handling clean and simple
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# A no-frills, beginner-friendly feedforward neural net
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)   # input layer to hidden
        self.fc2 = nn.Linear(128, 64)        # hidden layer
        self.fc3 = nn.Linear(64, 10)         # output layer (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)              # flattening the 28x28 image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                      # output is raw scores (we'll apply softmax if needed outside)
        return x

# Instantiating and pushing model to our chosen device
model = SimpleNN().to(device)

# Cross entropy loss works great for classification problems like MNIST
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Let's train the model (very basic loop for now)
epochs = 5
print("Training...")

for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("Finished training.")

# Saving model state_dict (weights only) to file
torch.save(model.state_dict(), "mnist_simple_nn.pth")

# Quick test run on one example to make sure things worked
data_iter = iter(test_loader)
test_image, test_label = next(data_iter)

# Push input to device and get prediction
model.eval()
test_image = test_image.to(device)
with torch.no_grad():
    output = model(test_image)
    predicted_label = torch.argmax(output).item()

print(f"Predicted label: {predicted_label}, Actual label: {test_label.item()}")

# Show the test image so we can visually verify the digit
plt.imshow(test_image.cpu().squeeze(), cmap="gray")
plt.title(f"Prediction: {predicted_label} | Actual: {test_label.item()}")
plt.axis("off")
plt.show()

# Optional: load model back up again just to confirm it all works
loaded_model = SimpleNN().to(device)
loaded_model.load_state_dict(torch.load("mnist_simple_nn.pth", map_location=device))
loaded_model.eval()
print("✅ Model reloaded — all keys matched successfully.")
