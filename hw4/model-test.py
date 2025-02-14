import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # LeNet-5 original structure:
        # Input: 1x28x28 (grayscale) -> convert to 32x32
        # C1: Conv2D -> 6x28x28
        # S2: AvgPool -> 6x14x14
        # C3: Conv2D -> 16x10x10
        # S4: AvgPool -> 16x5x5
        # C5: Conv2D -> 120x1x1 (fully connected as conv)
        # F6: 84 units fully connected
        # Output: 10 units fully connected (digits 0-9)

        self.C1 = nn.Conv2d(1, 6, kernel_size=5)   # (N,1,32,32)->(N,6,28,28)
        self.S2 = nn.AvgPool2d(2)                    # (N,6,28,28)->(N,6,14,14)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5)   # (N,6,14,14)->(N,16,10,10)
                                                       # pool -> (N,16,5,5)
        self.S4 = nn.AvgPool2d(2)
        # self.C5 = nn.Conv2d(16, 120, kernel_size=5)
        # self.flat = nn.Flatten()
        self.C5 = nn.Linear(16*5*5, 120)              # (N,16*5*5)->(N,120)
        self.F6 = nn.Linear(120, 84)                  # (N,120)->(N,84)
        self.output = nn.Linear(84, 10)                   # (N,84)->(N,10)

    def hyp_tan(self,x):
        # f(a) = Atan(Sa), where A=1.7159, S=(2/3)
        return (1.7159)*torch.tanh((2/3)*x)
    
    def forward(self, x):
        # print("Input shape:", x.shape)  # Input shape
        x = self.hyp_tan(self.C1(x))
        x = self.S2(x)
        x = self.hyp_tan(self.C3(x))
        x = self.S4(x)
        x = x.view(-1, 16*5*5)  # Flatten before FC layers
        x = self.hyp_tan(self.C5(x))
        x = self.hyp_tan(self.F6(x))
        x = self.output(x)  
        return x
    

if __name__ == "__main__":
    # Set up transform for MNIST: to tensor and normalize
    transform = transforms.Compose([\
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset from torchvision
    train_dataset = datasets.MNIST(root='./data.df_train', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data.df_test', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 1
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Test Accuracy:", correct / total *100)

    # extract weights between F6 to output
    weights = model.output.weight.data  # shape [10, 84]
    # reshape weights into 7x12 bitmaps
    bitmaps = weights.view(10, 7, 12)

    # Visualize the bitmaps
    for i, bitmap in enumerate(bitmaps):
        plt.figure()
        plt.subplot()
        plt.imshow(bitmap.cpu().numpy(), cmap='gray', vmin=weights.min(), vmax=weights.max())
        plt.title(f"Output[{i}] Connection Bitmap")
        plt.colorbar()
        plt.show()
    

    # Assuming `model`, `test_loader`, and `device` are already defined
    model.eval()

    # Initialize confusion matrix
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)

    # Dictionary to track the most confusing examples
    most_confusing_examples = {i: {'image': None, 'confidence': 0, 'predicted': None} for i in range(10)}

    # Collect predictions, true labels, and confidences
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # Get probabilities
            preds = torch.argmax(probs, dim=1)  # Get predictions

            # Update confusion matrix
            for true_label, predicted_label, confidence, image in zip(labels, preds, probs, images):
                confusion_matrix[true_label, predicted_label] += 1

                # Handle misclassified examples
                if true_label != predicted_label:
                    pred_conf = confidence[predicted_label].item()
                    # Check if this is the most confident misclassification for this true label
                    if pred_conf > most_confusing_examples[true_label.item()]['confidence']:
                        most_confusing_examples[true_label.item()] = {
                            'image': image.cpu(),
                            'confidence': pred_conf,
                            'predicted': predicted_label.item()
                        }

    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix.numpy(), cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.show()

    # Display the most confusing examples
    for true_label, example in most_confusing_examples.items():
        image = example['image']
        confidence = example['confidence']
        predicted_label = example['predicted']
        if image is not None:
            plt.figure()
            plt.imshow(image.squeeze().numpy(), cmap='gray')
            plt.title(f"True Label: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.4f}")
            plt.axis('off')
            plt.show()