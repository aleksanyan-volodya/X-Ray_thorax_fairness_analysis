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

def equalization(image: Image) -> Image:
    """
    Equalize the histogram of an image
    :param image: PIL image
    :return: PIL image
    """
    eq = ImageEnhance.Contrast(image).enhance(2)
    return eq

def transform_image(image: Image, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0, equalize: bool = False) -> Image:
    """
    Transform an image by applying rotation, brightness, saturation, noise and blur
    :param image: PIL image
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :param equalize: bool
    :return: PIL image
    """
    if rotation != 0:
        image = rotate_image(image, rotation)
    if brightness != 1.0:
        image = change_brightness(image, brightness)
    if noise != 0.0:
        image = add_noise(image, noise)
    if blur != 0.0:
        image = add_blur(image, blur)
    if equalize:
        image = equalization(image)
    return image


############################################


def create_transformed_image(image_path: str, transformed_image_path: str, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0, equalize: bool = False) -> None:
    """
    Add a transformed image to the dataset
    :param image_path: str
    :param transformed_image_path: str
    optional parameters
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :param equalize: bool
    :return: None
    """
    image = Image.open(image_path)
    transformed_image = transform_image(image, rotation, brightness, noise, blur, equalize)
    transformed_image.save(transformed_image_path)

# if to_csv_path is not precised, it will be the same as from_csv_path
def add_transformed_image_to_csv(from_csv_path: str, image_index: str, transformed_image_index: str, to_csv_path: str = None) -> None:
    """
    Add a transformed image to the dataset
    :param from_csv_path: str
    :param image_index: str
    :param transformed_image_index: str
    optional parameters
    :param to_csv_path: str
    :return: None
    """
    
    df_from = pd.read_csv(from_csv_path)
    image_info = df_from[df_from['Image Index'] == image_index].copy()
    image_info['Image Index'] = transformed_image_index
    
    # add the new image to the end of the csv
    df_to = pd.read_csv(to_csv_path) if to_csv_path != None else df_from
    df_to = pd.concat([df_to, image_info], ignore_index=True) 
    
    if to_csv_path is not None:
        df_to.to_csv(to_csv_path, index=False)
    else:
        df_to.to_csv(from_csv_path, index=False)

def add_new_image(from_image_path: str, from_csv_path: str, to_csv_path: str = None, to_image_path: str = None, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0, equalize: bool = False) -> None:
    """
    Add a new image to the dataset
    :param from_image_path: str
    :param from_csv_path: str
    oprional parameters
    :param to_csv_path: str
    :param to_image_path: str
    :param rotation: int
    :param brightness: float
    :param noise: float
    :param blur: float
    :param equalize: bool
    :return: None
    """
    splited_list = from_image_path.split('/')
    path, image_index = splited_list[:-1], splited_list[-1]
    
    if to_image_path is None:
        params_string = f"r{rotation}_b{brightness}_n{noise}_bl{blur}_eq{int(equalize)}"
        path = '/'.join(path)
        transformed_image_index = f"{image_index[:-4]}_{params_string}_transformed.png"
        transformed_image_path = f"{path}/{image_index[:-4]}_{params_string}_transformed.png"
    else:
        transformed_image_index = to_image_path.split('/')[-1]
        transformed_image_path = to_image_path

    create_transformed_image(from_image_path, transformed_image_path, rotation, brightness, noise, blur, equalize)
    add_transformed_image_to_csv(from_csv_path, image_index, transformed_image_index, to_csv_path=to_csv_path)


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


# ### try class
# ### including all functions above
# class ImageTransformation:
#     def __init__(self, image_path: str, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0, equalize: bool = False):
#         self.image_path = image_path
#         self.image = Image.open(image_path)

#         self.rotation = rotation
#         self.brightness = brightness
#         self.noise = noise
#         self.blur = blur
#         self.equalize = equalize

#         self.transformed_image = None
#         self.transformed_image_path = None


#     def __rotate_image(self) -> None:
#         """
#         Rotate an image by a given angle
#         """
#         self.transformed_image = self.image.rotate(self.angle)
#         return
    
#     def __change_brightness(self) -> None:
#         """
#         Change the brightness of an image
#         """
#         enhancer = ImageEnhance.Brightness(self.image)
#         self.transformed_image = enhancer.enhance(self.brightness)
#         return
    
#     def __add_noise(self) -> None:
#         """
#         Add noise to an image
#         """
#         np_image = np.array(self.image)
#         noise = np.random.normal(0, self.noise, np_image.shape)
#         noisy_image = np_image + noise
#         noisy_image_clipped = np.clip(noisy_image, 0, 255)
#         self.transformed_image = Image.fromarray(noisy_image_clipped.astype(np.uint8))
#         return
    
#     def __add_blur(self) -> None:
#         """
#         Add blur to an image
#         """
#         self.transformed_image = self.image.filter(ImageFilter.GaussianBlur(self.blur))
#         return
    
#     def __equalization(self) -> None:
#         """
#         Equalize the histogram of an image
#         """
#         eq = ImageEnhance.Contrast(self.image).enhance(2)
#         self.transformed_image = eq
#         return

#     def transform_image(self) -> None:
#         """
#         Transform an image by applying rotation, brightness, saturation, noise and blur
#         """ 

#         if self.rotation != 0:
#             self.__rotate_image()
#         if self.brightness != 1.0:
#             self.__change_brightness()
#         if self.noise != 0.0:
#             self.__add_noise()
#         if self.blur != 0.0:
#             self.__add_blur()
#         if self.equalize:
#             self.__equalization()
        
#         return

#     def save_transformed_image(self, transformed_image_path: str) -> None:
#         """
#         Save the transformed image to a file
#         :param transformed_image_path: str
#         """
#         self.transform_image()
#         if self.transformed_image is not None:
#             self.transformed_image.save(transformed_image_path)
#             self.transformed_image_path = transformed_image_path
#         else:
#             raise ValueError("Transformed image is None. Please transform the image before saving.")

#     def show_transformed_image(self) -> None:
#         """
#         Show the transformed image
#         """
#         if self.transformed_image is None:
#             self.transformed_image = self.image.copy()
        
#         plt.imshow(self.transformed_image)
#         plt.axis('off')
#         plt.show()
    
#     def add_transformed_image_csv(self, csv_path: str, image_index: str, transformed_image_index: str) -> None:
#         """
#         Add the transformed image to the csv file
#         :param csv_path: str
#         :param image_index: str
#         :param transformed_image_index: str
#         """
#         df = pd.read_csv(csv_path)
#         image_info = df[df['Image Index'] == image_index].copy()
#         image_info['Image Index'] = transformed_image_index
        
#         # add the new image to the end of the csv
#         df = pd.concat([df, image_info], ignore_index=True)
        
#         df.to_csv(csv_path, index=False)

# """
# def add_transformed_image_to_csv(from_csv_path: str, image_index: str, transformed_image_index: str, to_csv_path: str) -> None:

# def add_new_image(from_image_path: str, from_csv_path: str, to_csv_path: str = None, to_image_path: str = None, rotation: int = 0, brightness: float = 1.0, noise: float = 0.0, blur: float = 0.0, equalize: bool = False) -> None:

# def remove_image(image_path: str) -> None:


# def remove_transformed_data(csv_path: str) -> int:

# """