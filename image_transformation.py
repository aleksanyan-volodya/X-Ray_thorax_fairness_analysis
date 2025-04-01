# given an gray-scale image, this script will transform it into a new gray-scaled image with the same size, but different properties
# the properties are:
# - rotation
# - brightness
# - saturation
# - noise
# - blur

# Finally we will add the transformed image to the dataset

import os
import io
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def rotate_image(image: Image, angle: int) -> Image:
    """
    Rotate an image by a given angle
    :param image: PIL image
    :param angle: int
    :return: PIL image
    """
    return image.rotate(angle)

def change_brightness(image: Image, factor: float) -> Image:
    """
    Change the brightness of an image
    :param image: PIL image
    :param factor: float ]
    :return: PIL image
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def add_noise(image: Image, factor: float) -> Image:
    """
    Add noise to an image
    :param image: PIL image
    :param factor: float
    :return: PIL image
    """
    np_image = np.array(image)
    noise = np.random.normal(0, factor, np_image.shape)
    noisy_image = np_image + noise
    noisy_image_clipped = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image_clipped.astype(np.uint8))


def add_blur(image: Image, factor: float) -> Image:
    """
    Add blur to an image
    :param image: PIL image
    :param factor: float
    :return: PIL image
    """
    return image.filter(ImageFilter.GaussianBlur(factor))

def transform_image(image: Image, rotation: int, brightness: float, noise: float, blur: float) -> Image:
    """
    Transform an image by applying rotation, brightness, saturation, noise and blur
    :param image: PIL image
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :return: PIL image
    """
    image = rotate_image(image, rotation)
    image = change_brightness(image, brightness)
    image = add_noise(image, noise)
    image = add_blur(image, blur)
    return image

def add_transformed_image(image_path: str, transformed_image_path: str, rotation: int, brightness: float, noise: float, blur: float) -> None:
    """
    Add a transformed image to the dataset
    :param image_path: str
    :param transformed_image_path: str
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :return: None
    """
    image = Image.open(image_path)
    transformed_image = transform_image(image, rotation, brightness, noise, blur)
    transformed_image.save(transformed_image_path)