import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2, os, re, glob
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

label_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '#'}

MODEL_PATH = r"D:\Users\r2shaji\Downloads\best.pt"
model = YOLO(MODEL_PATH)

CROP_IMAGE_FOLDER = r"D:\Users\r2shaji\Downloads\ocr_28102028\train\crop_sharp"

class ClassifierNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ClassifierNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def extract_embeddings(img, embed_layers):
    
    embed_result = model.predict(img, embed=embed_layers)
    features = embed_result[0]
    return features


def pad_to_size(img, required_size=(256, 256)):

    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    
    original_width, original_height = img.size
    desired_width, desired_height = required_size

    pad_left = pad_right = pad_top = pad_bottom = 0

    if original_width < desired_width:
        total_pad_width = desired_width - original_width
        pad_left = total_pad_width // 2
        pad_right = total_pad_width - pad_left

    if original_height < desired_height:
        total_pad_height = desired_height - original_height
        pad_top = total_pad_height // 2
        pad_bottom = total_pad_height - pad_top

    padding = (pad_left, pad_top, pad_right, pad_bottom)  
    padded_img = transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

    numpy_img = np.array(padded_img)
    if len(numpy_img.shape) == 2:  
        numpy_img = np.expand_dims(numpy_img, axis=-1)

    return numpy_img


def prepare_image(image_tensor):

    image = pad_to_size(image_tensor, (256, 256))

    # crop_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # plt.imshow(crop_image)
    # plt.axis('off')  
    # plt.show()

    return image

def extract_label(image_path):
    match = re.search(r"___(\d+)", image_path)
    label = match.group(1) if match else None
    return int(label)

def load_image_features(embed_layers):

    image_paths = sorted(
    glob.glob(os.path.join(CROP_IMAGE_FOLDER, "*.jpg")) 
    )

    all_features = []
    true_labels = []

    for image_path in image_paths:
        image = Image.open(image_path)
        true_label = extract_label(image_path)
        image_bgr = prepare_image(image)
        extracted = extract_embeddings(image_bgr, embed_layers)
        
        if len(extracted)<1:
            continue
        
        all_features.append(extracted)
        true_labels.append(true_label)
        
    return np.array(all_features), np.array(true_labels)

def create_data_loaders(X, y, batch_size=32, train_split=0.8):

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):

    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return train_losses


def evaluate_model(model, test_loader):

    all_preds = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return all_labels, all_preds


def plot_confusion_matrix(all_labels, all_preds):

    class_labels = sorted(set(all_labels) | set(all_preds))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(37, 37))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title("Confusion Matrix")

    plt.savefig('test_confusion_matrix.png', bbox_inches='tight')


def plot_training_loss(train_losses):

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title("Epoch vs Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()

    plt.savefig(f'Training_Loss.png')

def main():
    torch.manual_seed(42)

    # Configuration
    embed_layers = [6, 15, 21]

    # Extract image features
    X, y = load_image_features(embed_layers)
    
    # Prepare data loaders for training and testing
    train_loader, test_loader = create_data_loaders(X, y, batch_size=32, train_split=0.8)
    
    # Define model parameters
    input_dim = 1152      
    hidden_dim = 256     
    num_classes = 37     
    model_classifier = ClassifierNet(input_dim, hidden_dim, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_classifier.parameters(), lr=0.001)
    num_epochs = 100
    
    # Train the model
    train_losses = train_model(model_classifier, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model on the test set
    all_labels, all_preds = evaluate_model(model_classifier, test_loader)

    mapped_labels = list(map(lambda x: label_names.get(x), all_labels))
    mapped_preds = list(map(lambda x: label_names.get(x), all_preds))
    
    # Plot the confusion matrix and training loss curve
    plot_confusion_matrix(mapped_labels, mapped_preds)
    plot_training_loss(train_losses)

    # save the model
    torch.save(model_classifier.state_dict(), "results/model_classifier.pth")
    print("Model saved as 'model_classifier.pth'")


if __name__ == "__main__":
    main()