import os
from glob import glob
from glog import logger
from torch.utils.data import Dataset
from data import aug
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import pandas as pd


class LPBlurDataset(Dataset):
    def __init__(self, opt):
        super(LPBlurDataset, self).__init__()
        self.opt = opt
        self.label_dir_path = os.path.join(opt.dataroot, opt.mode, 'label')
        self.files_a = os.path.join(opt.dataroot, opt.mode, 'blur')
        self.files_b = os.path.join(opt.dataroot, opt.mode, 'sharp')

        self.blur = glob(os.path.join(self.files_a, '*.jpg'))
        self.sharp = glob(os.path.join(self.files_b, '*.jpg'))
        assert len(self.blur) == len(self.sharp)

        if self.opt.mode == 'train':
            self.transform_fn = aug.get_transforms(size=(112, 224))
            self.transform_fn1 = aug.get_transforms(size=(56, 112))
            self.transform_fn2 = aug.get_transforms(size=(28, 56))
            self.transform_fn3 = aug.get_transforms(size=(14, 28))

        else:
            self.transform_fn = aug.get_transforms_fortest(size=(112, 224))
            self.transform_fn1 = aug.get_transforms_fortest(size=(56, 112))
            self.transform_fn2 = aug.get_transforms_fortest(size=(28, 56))
            self.transform_fn3 = aug.get_transforms_fortest(size=(14, 28))

        self.normalize_fn = aug.get_normalize()
        logger.info(f'Dataset has been created with {len(self.blur)} samples')

    def __len__(self):
        return len(self.blur)

    def __getitem__(self, idx):
        blur_image = Image.open(self.blur[idx])
        sharp_image = Image.open(self.sharp[idx])
        blur_image = np.array(blur_image)
        sharp_image = np.array(sharp_image)

        blur_image, sharp_image = self.transform_fn(blur_image, sharp_image)
        blur_image1, sharp_image1 = self.transform_fn1(blur_image, sharp_image)
        blur_image2, sharp_image2 = self.transform_fn2(blur_image, sharp_image)
        blur_image3, sharp_image3 = self.transform_fn3(blur_image, sharp_image)

        blur_image, sharp_image = self.normalize_fn(blur_image, sharp_image)
        blur_image1, sharp_image1 = self.normalize_fn(blur_image1, sharp_image1)
        blur_image2, sharp_image2 = self.normalize_fn(blur_image2, sharp_image2)
        blur_image3, sharp_image3 = self.normalize_fn(blur_image3, sharp_image3)
        blur_image = transforms.ToTensor()(blur_image)
        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image1 = transforms.ToTensor()(blur_image1)
        sharp_image1 = transforms.ToTensor()(sharp_image1)
        blur_image2 = transforms.ToTensor()(blur_image2)
        sharp_image2 = transforms.ToTensor()(sharp_image2)
        blur_image3 = transforms.ToTensor()(blur_image3)
        sharp_image3 = transforms.ToTensor()(sharp_image3)

        if self.opt.mode == 'train':
            plate_info = self.load_label(self.sharp[idx], self.label_dir_path)

            return {'A': blur_image, 'B': sharp_image, 'A_paths': self.blur[idx], 'B_paths': self.sharp[idx],
                    'A1': blur_image1, 'B1': sharp_image1, 'A2': blur_image2, 'B2': sharp_image2, 'A3': blur_image3,
                    'B3': sharp_image3, 'plate_info': plate_info}  # A: blur B: sharp

        else:
            return {'A': blur_image, 'B': sharp_image, 'A_paths': self.blur[idx], 'B_paths': self.sharp[idx],
                    'A1': blur_image1, 'B1': sharp_image1, 'A2': blur_image2, 'B2': sharp_image2, 'A3': blur_image3,
                    'B3': sharp_image3}

    def load_data(self):
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.num_threads))
        return dataloader
    
    def load_label(self, image_path, label_dir_path):
        label_file_name = os.path.basename(image_path).split(".")[0] + ".txt"
        label_file_path = os.path.join(label_dir_path,label_file_name)
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        lines= [line.rstrip() for line in lines]
        lines= [line.split() for line in lines]
        lines = np.array(lines).astype(float)
        lines = torch.from_numpy(lines)
        lines = sorted(lines, key=lambda x: x[1])
        sorted_labels = torch.tensor([t[0] for t in lines], dtype=torch.float64)
        sorted_boxes = [t[1:] for t in lines]
        plate_info = { "sorted_labels": sorted_labels, "sorted_boxes_xywhn":sorted_boxes}
        return plate_info


def create_dataset(opt):
    return LPBlurDataset(opt).load_data()






