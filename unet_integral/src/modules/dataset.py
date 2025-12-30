import cv2
from typing import List, Tuple, Optional
from torch.utils.data import Dataset


class SegmDataset(Dataset):
    """
        UNet Dataset class
    """
    def __init__(
        self,
        images_names: List,
        masks_names: Optional[List] = None,
        transforms=None,
    ):
        super().__init__()
        self.images_names = images_names
        self.masks_names = masks_names
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple:

        # Read image, mask
        image_path = self.images_names[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Apply augs
        if self.masks_names is not None:
            mask = cv2.imread(self.masks_names[idx])[:, :, :1] / 255
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image'].float()
            mask = transformed['mask'].float()
            return image, image_path, (height, width), mask
        else:
            transformed = self.transforms(image=image)
            image = transformed['image'].float()
            return image, image_path, (height, width), image.shape[1:]

    def __len__(self) -> int:
        return len(self.images_names)
