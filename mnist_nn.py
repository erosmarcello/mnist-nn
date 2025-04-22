# Eros Marcello | Simple NN for MNIST digits. Classical baseline to scale for Quantum ML EEG classifier.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # For visualization later

# Use GPU if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading ---
# Standard MNIST transform: ToTensor scales to [0, 1]
transform = transforms.ToTensor()
# Load datasets
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# DataLoaders. Batch 64 for training, shuffle enabled. Test batch 1 for single-image check later
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# --- Model Definition ---
# Basic Feedforward Network | 784 -> 128 -> 64 -> 10.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__() # Init parent class
        self.fc1 = nn.Linear(28 * 28, 128)   # Input layer (flattened image) to hidden 1
        self.fc2 = nn.Linear(128, 64)        # Hidden 1 to hidden 2
        self.fc3 = nn.Linear(64, 10)         # Hidden 2 to output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)              # Flatten image batch
        x = torch.relu(self.fc1(x))          # Activation after fc1
        x = torch.relu(self.fc2(x))          # Activation after fc2
        x = self.fc3(x)                      # Output logits (raw scores). Softmax done by loss
        return x

# Instantiate model and mov to target device
model = SimpleNN().to(device)

# --- Training Components ---
# CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()
# Adam optimizer lr=0.001 is a decent default
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 5 # of training passes
print("Training...")

model.train() # Set model to training 
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader: # Iterate over batches
        images, labels = images.to(device), labels.to(device) # Data to device

        optimizer.zero_grad()   # Reset gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels) # Calculate loss
        loss.backward()         # Backpropagationnnnnn
        optimizer.step()        # Update weights

        total_loss += loss.item() # Accumulate batch loss.

    avg_loss = total_loss / len(train_loader) # Average loss for the epoch.
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}") # Log progress.

print("Finished training.")

# --- Save Model State ---
# Save only the model weights (state_dict). Efficient.
torch.save(model.state_dict(), "mnist_simple_nn.pth")
print("Saved model state_dict to mnist_simple_nn.pth")

# --- Quick Inference Test ---
# Verify on a single test image.
data_iter = iter(test_loader) # Get iterator.
test_image, test_label = next(data_iter) # Get first sample.

# Set model to evaluation mode. Important for layers like Dropout/BatchNorm if used.
model.eval()
test_image = test_image.to(device) # Image to device.
with torch.no_grad(): # Disable gradient calculation for inference.
    output = model(test_image) # Get model output
    predicted_label = torch.argmax(output).item() # Get predicted class index

print(f"Predicted label: {predicted_label}, Actual label: {test_label.item()}")

# Display the test image and result
plt.imshow(test_image.cpu().squeeze(), cmap="gray") # Plot image (CPU, remove batch dim)
plt.title(f"Prediction: {predicted_label} | Actual: {test_label.item()}") # Add title
plt.axis("off") # Clean axes
plt.show()

# --- Reload Model Check ---
# Optional step here but this Verifies if the fuckin loading works
loaded_model = SimpleNN().to(device) # Create a new model instance.
loaded_model.load_state_dict(torch.load("mnist_simple_nn.pth", map_location=device)) # Load weights
loaded_model.eval() # Set to eval
print("✅ Model reloaded — all keys matched successfully.") 
