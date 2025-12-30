import importlib
import typing as tp

import cv2
import numpy as np


def load_object(obj_path: str, default_obj_path: str = '') -> tp.Any:
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def read_rgb_img(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Image does not exist: {img_path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
