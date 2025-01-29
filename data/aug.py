import albumentations as albu
import torch
from torchvision import transforms


def get_transforms(size):
    augs = {'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                     albu.ShiftScaleRotate(always_apply=True),
                                     albu.Transpose(always_apply=True),
                                     albu.OpticalDistortion(always_apply=True),
                                     albu.ElasticTransform(always_apply=True),
                                     ])
            }

    aug_fn = augs['geometric']
    crop_fn = {'random': albu.RandomCrop(height=size[0], width=size[1], always_apply=True),

               'center': albu.CenterCrop(height=size[0], width=size[1], always_apply=True)}['random']

    effect = albu.OneOf([albu.MotionBlur(blur_limit=21, always_apply=True),
                         albu.RandomRain(always_apply=True),
                         albu.RandomFog(always_apply=True),
                         albu.RandomSnow(always_apply=True)])
    motion_blur = albu.MotionBlur(blur_limit=55, always_apply=True)

    resize = albu.Resize(height=size[0], width=size[1])

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

    pipforblur = albu.Compose([effect])

    def process(a, b):
        f = pipforblur(image=a)
        r = pipeline(image=f['image'], target=b)
        return r['image'], r['target']

    return process


def get_transforms_fortest(size):
    resize = albu.Resize(height=size[0], width=size[1])

    effect = albu.OneOf([albu.MotionBlur(always_apply=True),
                         albu.RandomRain(always_apply=True),
                         albu.RandomFog(always_apply=True),
                         albu.RandomSnow(always_apply=True)])
    motion_blur = albu.MotionBlur(blur_limit=51, always_apply=True)

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process

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

    return padded_img
