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
import torchvision.ops as ops
import seaborn as sns
from PIL import Image
import matplotlib.patches as patches
from pathlib import Path
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMAGE_FOLDER = r"D:\Users\r2shaji\Downloads\ocr_28102028\train\sharp"
LABEL_FOLDER = r"D:\Users\r2shaji\Downloads\ocr_28102028\train\labels"
model = YOLO(r"D:\Users\r2shaji\Downloads\best.pt")

label_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '#'}

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

def xywhn_to_xyxy(detection, height, width):
    cx, cy, w, h = detection
        
    x1 = int((cx - w / 2) * width)
    y1 = int((cy - h / 2) * height)
    x2 = int((cx + w / 2) * width)
    y2 = int((cy + h / 2) * height)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return x1, y1, x2, y2

def load_label(image_path):
        
    label_file_name = os.path.basename(image_path).replace('.jpg', '.txt')
    label_file_path = os.path.join(LABEL_FOLDER,label_file_name)
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    lines= [line.rstrip() for line in lines]
    lines= [line.split() for line in lines]
    lines = np.array(lines).astype(float)
    lines = torch.from_numpy(lines)
    lines = sorted(lines, key=lambda x: x[1])
    sorted_labels = torch.tensor([t[0] for t in lines],  dtype=torch.float)
    sorted_labels = sorted_labels.long()
    sorted_boxes = [t[1:] for t in lines]
    ground_truth = { "sorted_labels": sorted_labels, "sorted_boxes_xywhn":sorted_boxes}
    return ground_truth

def plot_and_save_cropped_feature_map(cropped_feats, plot_image_name):

    B, C, H, W = cropped_feats.shape
    if B != 1:
        print("Warning: This function expects a single batch (B=1). Taking the first element.")

    fm = cropped_feats[0]

    n = C  
    rows = math.ceil(n / 8)
    cols = min(n, 8)

    fig, ax = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    ax = ax.ravel() if n > 1 else [ax]  

    for i in range(n):
        ax[i].axis('off')
        ax[i].imshow(fm[i].cpu().numpy(), cmap='gray')
        ax[i].set_title(f"Channel {i}")

    for j in range(n, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    fig.savefig(plot_image_name, dpi=fig.dpi)
    plt.close(fig)


def crop_features(feature_map, bbox, output_size=(10, 10)):

    _, _, height, width = feature_map.shape

    xmin_feat, ymin_feat, xmax_feat, ymax_feat = xywhn_to_xyxy(bbox, height, width)

    boxes = torch.tensor([[0, xmin_feat, ymin_feat, xmax_feat, ymax_feat]], device=feature_map.device,  dtype=torch.float)

    # cropped_feature = ops.roi_align(feature_map, boxes, output_size=output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    cropped_feature = feature_map[:, :, ymin_feat:ymax_feat, xmin_feat:xmax_feat]
    _, _, cropped_feature_height, cropped_feature_width = cropped_feature.shape
    cropped_feature_height, cropped_feature_width = int(cropped_feature_height), int(cropped_feature_width)
    
    print("cropped_feature_height, cropped_feature_width",cropped_feature_height, cropped_feature_width)
    if cropped_feature_width == 0 or cropped_feature_height == 0:
        if cropped_feature_width == 0:
            xmin_feat = xmin_feat-1 if xmin_feat>0 else 0
            xmax_feat = xmax_feat+1 if xmax_feat<width else width
        if cropped_feature_height == 0:
            ymin_feat = ymin_feat-1 if ymin_feat>0 else 0
            ymax_feat = ymax_feat+1 if ymax_feat<height else height
        cropped_feature = feature_map[:, :, ymin_feat:ymax_feat, xmin_feat:xmax_feat]
    
    return cropped_feature

def feature_visualization(x, im_ht, im_wid, bbox, stage, n=4, save_dir=Path("feature_plots")):
     print("stage",stage)

     if isinstance(x, torch.Tensor):
        
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"{save_dir}/{stage}_features.png"

            # Chunk the tensor into individual channel blocks
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)  # number of channels to plot

            # Create subplots; here, we use 8 columns.
            fig, ax = plt.subplots(math.ceil(n / 4), 4, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            for i in range(n):
                print("blocks[i]",blocks[i].shape)
                ax[i].imshow(blocks[i].squeeze())

                if bbox is not None:
                    for each_bbox in bbox:
                        xmin, ymin, xmax, ymax = xywhn_to_xyxy(each_bbox, height, width)
                        rect_width = xmax - xmin
                        rect_height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), rect_width, rect_height,
                                                linewidth=1, edgecolor='r', facecolor='none')
                        ax[i].add_patch(rect)
                
                ax[i].axis("off")

            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()


def load_image_features(embed_layers= [1,2,3,4]):

    image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) 
    )

    all_features = []
    true_labels = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        ground_truth = load_label(image_path)
        true_boxes = ground_truth["sorted_boxes_xywhn"]
        im_wid, im_ht = image.width, image.height

        

        to_tensor = transforms.ToTensor()
        image = to_tensor(image).unsqueeze(0)

        results = model.predict(image_path, embed=embed_layers) 
        cropped_char_features = [[] for _ in range(len(true_boxes))]

        k=0
        for diff_layer_feature in results:
            k+=1
            # feature_visualization(diff_layer_feature, im_ht, im_wid, true_boxes, f"{image_name}_stage_{k}")
            for i, bbox in enumerate(true_boxes):
                label = label_names[ground_truth["sorted_labels"][i].item()]
                # plot_image_name = f'feature_crops/{image_name}_{i}_{label}.png'
                cropped_feats = crop_features(diff_layer_feature, bbox)
                # plot_and_save_cropped_feature_map(cropped_feats, plot_image_name)
                print("cropped_feats before pooling shape",cropped_feats.shape)
                # print("cropped_feats before pooling",cropped_feats)
                cropped_feats = nn.functional.adaptive_avg_pool2d(cropped_feats, (1, 1)).squeeze(-1).squeeze(-1)
                print("cropped_feats after pooling shape",cropped_feats.shape)
                if torch.isnan(cropped_feats).any():
                    print("Tensor has NaN values.", cropped_feats)
                    print("image name",image_name)
                print("label",label)
                cropped_char_features[i].append(cropped_feats)

        # for i, bbox in enumerate(true_boxes):
        #     cropped_image = crop_features(image, bbox, im_ht, im_wid)
        #     flattened = nn.functional.adaptive_avg_pool2d(cropped_image, (1, 1)).squeeze(-1).squeeze(-1)
        #     # print("flattened", flattened.shape)
        #     cropped_char_features[i].append(flattened)


        for i, char_embeddings in enumerate(cropped_char_features):
            cropped_char_features[i] = torch.unbind(torch.cat(char_embeddings, 1), dim=0)[0].squeeze(0)
            # print("cropped_feats",cropped_char_features[i])
            print("cropped_feats",cropped_char_features[i].shape)
            all_features.append(cropped_char_features[i])
            true_labels.append(ground_truth["sorted_labels"][i].item())

    return np.array(all_features), np.array(true_labels)

def create_data_loaders(X, y, batch_size=32, train_split=0.8):
    print("length of X",len(X))
    print("shape of X",X.shape)
    print("length of y",len(y))
    print("shape of y",y.shape)
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
    
    cm = confusion_matrix(all_labels, all_preds, labels=class_labels)
    print("Confusion Matrix:\n", cm)

    sns.set_theme(rc={'figure.figsize': (20, 20)})
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")


    # plt.figure(figsize=(20, 20))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    # disp.plot(cmap=plt.cm.Blues)
    
    plt.title("Confusion Matrix")

    plt.savefig('test_confusion_matrix_feature_crop.png')

def plot_training_loss(train_losses):

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title("Epoch vs Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()

    plt.savefig(f'training_Loss_feature_crop.png')

def main():
    torch.manual_seed(42)

    # Configuration
    embed_layers = [1,2,3,4]

    X, y = load_image_features(embed_layers)
    
    # Prepare data loaders for training and testing
    train_loader, test_loader = create_data_loaders(X, y, batch_size=32, train_split=0.8)
    
    # Define model parameters
    input_dim = 576      
    hidden_dim = 256     
    num_classes = 37     
    model_classifier = ClassifierNet(input_dim, hidden_dim, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_classifier.parameters(), lr=0.001)
    num_epochs = 120
    
    # Train the model
    train_losses = train_model(model_classifier, train_loader, criterion, optimizer, num_epochs)

    # model_classifier.load_state_dict(torch.load("results/feature_crop_classifier.pth"))
    
    # Evaluate the model on the test set
    all_labels, all_preds = evaluate_model(model_classifier, test_loader)

    mapped_labels = list(map(lambda x: label_names.get(x), all_labels))
    mapped_preds = list(map(lambda x: label_names.get(x), all_preds))
    
    # Plot the confusion matrix and training loss curve
    plot_confusion_matrix(mapped_labels, mapped_preds)
    plot_training_loss(train_losses)

    # save the model
    torch.save(model_classifier.state_dict(), "results/feature_crop_classifier.pth")
    print("Model saved as 'feature_crop_classifier.pth'")


if __name__ == "__main__":
    main()
