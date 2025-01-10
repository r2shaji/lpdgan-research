import torch.nn as nn
import torch
from ultralytics.utils.metrics import bbox_iou
import os
import numpy
from ultralytics import YOLO
import cv2
from PIL import Image
import torchvision.transforms as T
import os

class CELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(CELoss, self).__init__()
        self.eps = eps

    def select_indices_target_boxes(self, pred_boxes, true_boxes, ciou_threshold=0.3):

        matched_indices = {}
        if len(pred_boxes) == 0:
            return matched_indices

        pred_boxes = torch.stack(pred_boxes)
        true_boxes = torch.stack(true_boxes)
        pred_boxes = pred_boxes.float().to(pred_boxes.device)
        true_boxes = true_boxes.float().to(true_boxes.device)

        for idx, pred_box in enumerate(pred_boxes):
            ciou_scores = bbox_iou(pred_box,true_boxes, xywh=True, CIoU=True)
            best_ciou= ciou_scores.max()
            matched_idx = torch.argmax(ciou_scores)
            if best_ciou > ciou_threshold:
                matched_indices[idx] = matched_idx.item()

        return matched_indices
    
    def sort_bbox(self, bbox):
        return sorted(bbox, key=lambda x: x[0])
    
    def sort_class_log_prob(self,boxes, log_prob):
        x1_coordinates = boxes[:, 0]
        sorted_indices = torch.argsort(x1_coordinates)
        sorted_log_prob = log_prob[0][sorted_indices]
        sorted_log_prob = torch.log(sorted_log_prob.clamp(min=1e-12))
        return sorted_log_prob
    
    def find_corresponding_correct_label(self,matched_indices,sorted_labels):
        relevant_indices = list(matched_indices.values())
        relevant_values = sorted_labels[relevant_indices]
        return relevant_values
    
    def select_relevant_log_prob(self,sorted_class_log_prob,matched_indices):
        relevant_indices = list(matched_indices.keys())
        relevant_values = sorted_class_log_prob[relevant_indices]
        return relevant_values

    def get_loss(self, log_probs, true_cls, epsilon, loss_cls_max=3.58):
        max_loss = 1
        if len(log_probs)<1 or len(true_cls)<1:
            return max_loss * len(true_cls)

        ce_loss = nn.CrossEntropyLoss(reduction='mean')
        loss_cls = ce_loss(log_probs, true_cls)
        loss_cls = min(loss_cls.item() / loss_cls_max, 1.0)
        return loss_cls
    
    def __call__(self, predicted_results, plate_info, epsilon=1e-9):
        sorted_class_log_prob = self.sort_class_log_prob(predicted_results.boxes.xywh, predicted_results.class_logits)
        sorted_fake_B_PlateNum = self.sort_bbox(predicted_results.boxes.xywhn)
        print("sorted_class_log_prob",sorted_class_log_prob)
        matched_indices = self.select_indices_target_boxes(sorted_fake_B_PlateNum, plate_info["sorted_boxes_xywhn"])
        mapped_labels = self.find_corresponding_correct_label(matched_indices,plate_info["sorted_labels"])
        mapped_labels = mapped_labels.long()
        relevant_log_prob = self.select_relevant_log_prob(sorted_class_log_prob,matched_indices)
        print("mapped_labels",mapped_labels)
        return self.get_loss(relevant_log_prob, mapped_labels, epsilon)
    



def load_label(image_path, label_dir_path):
    label_file_name = os.path.basename(image_path).split(".")[0] + ".txt"
    label_file_path = os.path.join(label_dir_path,label_file_name)
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    lines= [line.rstrip() for line in lines]
    lines= [line.split() for line in lines]
    lines = numpy.array(lines).astype(float)
    lines = torch.from_numpy(lines)
    lines = sorted(lines, key=lambda x: x[1])
    sorted_labels = torch.tensor([t[0] for t in lines], dtype=torch.float64)
    sorted_boxes = [t[1:] for t in lines]
    plate_info = { "sorted_labels": sorted_labels, "sorted_boxes_xywhn":sorted_boxes}
    return plate_info



if __name__=="__main__":
    model = YOLO(r"D:\Users\r2shaji\Downloads\ocr_detection_model_v7_sz512_20241102\ocr_detection_model_v7_sz512_20241102.onnx")  
    image = cv2.imread(r"D:\Users\r2shaji\Downloads\lpdata\ocr_merged\train\sharp\2a602f33-112d-4003-9580-cdb52dbdc245_52_737_gt_3281721_roi_3281721_151.jpg")
    fake_B_PlateNum = model.predict(image)

    plate_info = load_label(r"D:\Users\r2shaji\Downloads\lpdata\ocr_merged\train\sharp\2a602f33-112d-4003-9580-cdb52dbdc245_52_737_gt_3281721_roi_3281721_151.jpg",r"D:\Users\r2shaji\Downloads\lpdata\ocr_merged\train\label")
    ceLoss = CELoss()
    print(ceLoss(fake_B_PlateNum[0],plate_info))