""" This file takes code from tensorflow_example, Watermarking.py, get_data_set.py
The goal is to
1. download a set of jpg images
2. generate a second set of jpg images which are watermarked
3. convert from JPG to a Dataset usable by tensorflow NN's
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import random
import shutil
import IPython.display as display
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import time
import os
# from . import watermarking as WM

tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

#################################################################################
#################### DOWLOAD A SET OF JPG IMAGES ################################
#################################################################################
"""Import Stuff Created In This Section:

    data_root: a Path variable describing where a shit load of jpg images are stored
    all_image_paths: a list of paths that refer to a shit load of jpg images
    all_image_labels: a list of the categories that correspond to each of the images in all_image_paths
    my_attributions: a dictionary of things that describe the jpg images
    label_to_index: a dictionary of possible categories for these images

"""

# download the file
data_root_orig = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname='flower_photos', untar=True)

# save the path where the files were saved
data_root = pathlib.Path(data_root_orig)
print("JPG images downloaded to the following location: ", data_root)

# generate a list of all the files we just downloaded
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

# shuffle the list of jpg paths
random.shuffle(all_image_paths)

# process the LICENSE.txt file, which contains all the info we need to know about the images.
my_attributions = (data_root / "LICENSE.txt").open(encoding='utf-8').readlines()[4:]
my_attributions = [line.split(' CC-BY') for line in my_attributions]
my_attributions = dict(my_attributions)

"""This function parses out a caption for an image given its path and the overall attribtuions dict"""


def caption_image(image_path, attributions):
    # added as_posix() to this line, because windows path doesn't match LICENSE.txt format.
    image_rel = pathlib.Path(image_path).relative_to(data_root).as_posix()
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


# #
# # for n in range(3):
# #   image_path = random.choice(all_image_paths)
# #   display.display(display.Image(image_path))
# #   print(caption_image(image_path))
# #   print()


# determine the text for image label possibilities, then assign numbers to those options in a dict
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print("The label_to_index dict is: ", label_to_index)

# create a list of each images label, corresponding to all_image_paths
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

#################################################################################
#################### GENERATE A SET OF WATERMARKED IMAGES #######################
#################################################################################
"""Import Stuff Created In This Section:

    watermarked_root: a Path variable describing where a shit load of WATERMARKED jpg images are stored
    all_watermarked_paths: a list of paths that refer to a shit load of WATERMARKED jpg images
    
    The following variables still correspond to all_watermarked_paths:
    all_image_labels: a list of the categories that correspond to each of the images in all_image_paths
    my_attributions: a dictionary of things that describe the jpg images
    label_to_index: a dictionary of possible categories for these images

"""


pwd = os.path.dirname(os.path.realpath(__file__)) + "//"
FONT_PATH = pwd + "current_test_data//BERNHC.TTF"
WATERMARK_TEXT = "SHEETZ"


def watermark_text(input_image_path, output_image_path, text, pos):
    photo = Image.open(input_image_path)

    # make the image editable
    drawing = ImageDraw.Draw(photo)

    black = (3, 8, 12)
    font = ImageFont.truetype(FONT_PATH, size=50)
    drawing.text(pos, text, fill=black, font=font)
    # photo.show()
    photo.save(output_image_path)


# create a new directory path string to house the watermarked images (does not actually create the dir)
watermarked_root = data_root.with_name(str(data_root.name)+"_watermarked")

# blank list for strings indicating paths to watermarked photos, in the same order as all_image_paths
all_watermarked_paths = []


print("Creating Watermarked Directory....")
for img_path in all_image_paths:
    p = pathlib.Path(img_path)
    output_path = watermarked_root/p.parent.name/p.name
    watermark_text(
        img_path,
        output_path,
        text=WATERMARK_TEXT,
        pos=(0, 0)
    )
    all_watermarked_paths += [str(output_path)]

#################################################################################
#################### CONVERT FROM JPG TO DATASET ################################
#################################################################################


#################################################################################
################### END OF FILE #################################################
#################################################################################
