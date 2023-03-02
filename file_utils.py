import glob
import os

from pandas.core.common import flatten


def get_image_path(path: str):
    images = [glob.glob(f"{data_path}/*") for data_path in glob.glob(f"{path}/*")]
    return list(flatten(images))


def get_image_label(image: str) -> str:
    return image.split("/")[-2]


def find_classes(directory: str):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
