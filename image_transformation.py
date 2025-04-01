# -*- coding: utf-8 -*-

import os
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

def transform_image(image: Image, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0) -> Image:
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


############################################


def create_transformed_image(image_path: str, transformed_image_path: str, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0) -> None:
    """
    Add a transformed image to the dataset
    :param image_path: str
    :param transformed_image_path: str
    optional parameters
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :return: None
    """
    image = Image.open(image_path)
    transformed_image = transform_image(image, rotation, brightness, noise, blur)
    transformed_image.save(transformed_image_path)

def add_transformed_image_to_csv(csv_path: str, image_index: str, transformed_image_index: str) -> None:
    """
    Add a transformed image to the dataset
    :param csv_path: str
    :param image_index: str
    :param transformed_image_index: str
    """
    
    df = pd.read_csv(csv_path)
    image_info = df[df['Image Index'] == image_index].copy()
    image_info['Image Index'] = transformed_image_index

    # add the new image to the end of the csv
    df = pd.concat([df, image_info], ignore_index=True)
    df.to_csv(csv_path, index=False)

def add_new_image(image_path: str, csv_path: str, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0) -> None:
    """
    Add a new image to the dataset
    :param image_path: str
    :param csv_path: str
    oprional parameters
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :return: None
    """
    params_string = f"r{rotation}_b{brightness}_n{noise}_bl{blur}"

    splited_list = image_path.split('/')
    path, image_index = splited_list[:-1], splited_list[-1]
    path = '/'.join(path)
    
    transformed_image_path = f"{path}/{image_index[:-4]}_{params_string}_transformed.png"

    create_transformed_image(image_path, transformed_image_path, rotation, brightness, noise, blur)
    add_transformed_image_to_csv(csv_path, image_index, transformed_image_path)


###########################################


def remove_image(image_path: str) -> None:
    """
    Remove an image from the dataset without using it
    :param image_path: str
    :return: None
    """
    if os.path.exists(image_path):
        os.remove(image_path)
    else:
        raise FileNotFoundError(f"Image {image_path} not found")


def remove_transformed_data(csv_path: str) -> int:
    """
    Remove a transformed image from the dataset
    :param csv_path: str
    :return: number of transformed images remaining
    """

    df = pd.read_csv(csv_path)
    df_transformed = df[df['Image Index'].str.contains('_transformed.png')]

    for index, row in df_transformed.iterrows():
        transform_image_path = row['Image Index']
        try:
            remove_image(transform_image_path)
            df = df.drop(index)
        except Exception as e:
            print(f"Error removing image {transform_image_path}: {e}")

    df.to_csv(csv_path, index=False)
    
    return len(df[df['Image Index'].str.contains('_transformed.png')])