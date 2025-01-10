import albumentations as albu
import os
import cv2
import numpy as np

# transform = albu.OneOf([albu.MotionBlur(always_apply=True),
#                          albu.RandomRain(always_apply=True),
#                          albu.RandomFog(always_apply=True),
#                          albu.RandomSnow(always_apply=True)])

transform = albu.Compose([
    albu.MotionBlur(blur_limit=35, p=1.0),
    # albu.RandomRain(always_apply=True),
    # albu.RandomFog(always_apply=True),
    # albu.RandomSnow(always_apply=True)
    ])


def create_blur_folder(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):

            image = cv2.imread(input_path)

            transformed = transform(image=image)
            blurred_image = transformed['image']

            cv2.imwrite(output_path, blurred_image)
            print(f"Processed and saved: {output_path}")

input_folder = r"D:\Users\r2shaji\Downloads\lpdata\ocr_merged\train\sharp"
output_folder = r"D:\Users\r2shaji\Downloads\lpdata\ocr_merged\train\blur1"


create_blur_folder(input_folder, output_folder)