from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches
import os
import cv2


def load_label(image_path):
        LABEL_FOLDER = r"D:\Users\r2shaji\Downloads\ocr_28102028\train\labels"
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
        plate_info = { "sorted_labels": sorted_labels, "sorted_boxes_xywhn":sorted_boxes}
        return plate_info

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

# Load model
model = YOLO(r"D:\Users\r2shaji\Downloads\best.pt")

image_path = r"D:\Users\r2shaji\Downloads\ocr_28102028\train\sharp\trucknumber-1727795809120.jpg"

results = model.predict(image_path, embed=[6,15,21]) 
features1 = results[1]

print("features1",results[0].shape)
print("features2",results[1].shape)
print("features3",results[2].shape)

# features1 torch.Size([1, 384, 12, 32])
# features2 torch.Size([1, 192, 24, 64])
# features3 torch.Size([1, 576, 6, 16])

# features1 torch.Size([1, 384, 22, 32])
# features2 torch.Size([1, 192, 44, 64])
# features3 torch.Size([1, 576, 11, 16])

def feature_visualization(x, im_ht, im_wid, bbox, stage, n=32, save_dir=Path("feature_plots")):
     print("stage",stage)

     if isinstance(x, torch.Tensor):
        
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"{save_dir}/stage{stage}_features.png"

            # Chunk the tensor into individual channel blocks
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)  # number of channels to plot

            # Create subplots; here, we use 8 columns.
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
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


image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image from path: {image_path}")

height, width, _ = image.shape
bbox = load_label(image_path)
feature_visualization(results[0], height, width, bbox["sorted_boxes_xywhn"], 6)
feature_visualization(results[1], height, width, bbox["sorted_boxes_xywhn"], 15)
feature_visualization(results[2], height, width, bbox["sorted_boxes_xywhn"], 21)