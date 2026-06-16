import albumentations as albu
import cv2

def get_train_transforms(
    width: int = None,
    height: int = None,
    with_keypoints: bool = False,
) -> albu.BaseCompose:
    transforms = [
                albu.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                albu.RandomBrightnessContrast(p=0.3),
                albu.GaussNoise(p=0.3),
                albu.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225),
                               max_pixel_value=255.0,
                               p=1.0),
                albu.ToTensorV2(transpose_mask=True)
            ]

    kp_params = None
    if with_keypoints:
        kp_params = albu.KeypointParams(format='xy', remove_invisible=False)

    return albu.Compose(transforms, keypoint_params=kp_params)

def get_valid_transforms(
    width: int = None,
    height: int = None,
    with_keypoints: bool = False,
) -> albu.BaseCompose:
    transforms = [
                albu.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0,
                       p=1.0),
                albu.ToTensorV2(transpose_mask=True)
            ]

    kp_params = None
    if with_keypoints:
        kp_params = albu.KeypointParams(format='xy', remove_invisible=False)

    return albu.Compose(transforms, keypoint_params=kp_params)

