
import PIL
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torchvision.transforms as T
import os

# torch.transforms



# Gausian Blur
gausian_blur_transformation_13 = T.GaussianBlur(kernel_size=(7, 13), sigma=(6, 9))
gausian_blur_transformation_56 = T.GaussianBlur(kernel_size=(7, 13), sigma=(5, 8))


# Gausian Noise

def addnoise(input_image, noise_factor=0.3):
    inputs = T.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0, 1.)
    output_image = T.ToPILImage()
    image = output_image(noisy)
    return image


# Colour Jitter

colour_jitter_transformation_1 = T.ColorJitter(brightness=(0.5, 1.5), contrast=(3), saturation=(0.3, 1.5),
                                               hue=(-0.1, 0.1))

colour_jitter_transformation_2 = T.ColorJitter(brightness=(0.7), contrast=(6), saturation=(0.9), hue=(-0.1, 0.1))

colour_jitter_transformation_3 = T.ColorJitter(brightness=(0.5, 1.5), contrast=(2), saturation=(1.4), hue=(-0.1, 0.5))



# Main function that calls all the above functions to create 11 augmented images from one image

def augment_image(img_path):
    # orig_image
    orig_img = Image.open(Path(img_path))

    # Gausian Blur

    gausian_blurred_image_13_image = gausian_blur_transformation_13(orig_img)
    # gausian_blurred_image_13_image.show()

    gausian_blurred_image_56_image = gausian_blur_transformation_56(orig_img)
    # gausian_blurred_image_56_image.show()

    # Gausian Noise

    gausian_image_3 = addnoise(orig_img)

    # gausian_image_3.show()

    gausian_image_6 = addnoise(orig_img, 0.6)

    # gausian_image_6.show()

    gausian_image_9 = addnoise(orig_img, 0.9)

    # gausian_image_9.show()

    # Color Jitter

    colour_jitter_image_1 = colour_jitter_transformation_1(orig_img)

    # colour_jitter_image_1.show()

    colour_jitter_image_2 = colour_jitter_transformation_2(orig_img)

    # colour_jitter_image_2.show()

    colour_jitter_image_3 = colour_jitter_transformation_3(orig_img)

    # colour_jitter_image_3.show()

    return [orig_img,
            gausian_blurred_image_13_image, gausian_blurred_image_56_image, gausian_image_3, gausian_image_6,
            gausian_image_9, colour_jitter_image_1, colour_jitter_image_2, colour_jitter_image_3]


# augmented_images = augment_image(orig_img_path)

def creating_file_with_augmented_images(file_path_master_dataset, file_path_augmented_images):
    master_dataset_folder = file_path_master_dataset
    files_in_master_dataset = os.listdir(file_path_master_dataset)
    augmented_images_folder = file_path_augmented_images

    counter = 0

    for element in files_in_master_dataset:
        os.mkdir(f"{augmented_images_folder}/{element}")
        images_in_folder = os.listdir(f"{master_dataset_folder}/{element}")
        counter = counter + 1
        counter2 = 0
        for image in images_in_folder:
            required_images = augment_image(f"{master_dataset_folder}/{element}/{image}")
            counter2 = counter2 + 1
            counter3 = 0
            for augmented_image in required_images:
                counter3 = counter3 + 1
                augmented_image = augmented_image.save(
                    f"{augmented_images_folder}/{element}/{counter}_{counter2}_{counter3}_{image}")

# augmented dataset path
augmented_dataset = "/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/SpeckleRobotDatasetAug14"

# master dataset path
master_dataset = "/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/SpeckleRobotDataset_Upgraded"

# run the program

creating_file_with_augmented_images(master_dataset, augmented_dataset)
