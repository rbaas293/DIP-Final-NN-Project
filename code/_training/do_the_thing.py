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
    # watermark_text(
        # img_path,
        # output_path,
        # text=WATERMARK_TEXT,
        # pos=(0, 0)
    # )
    all_watermarked_paths += [str(output_path)]


#################################################################################
#################### CONVERT FROM JPG TO DATASET ################################
#################################################################################


def load_and_preprocess_image(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor=preprocess_image(img_raw)
    return img_tensor


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def build_from_list(paths, labels):
    # Build a tf.data.Dataset
    print("\n\nBuilding A Dataset...")
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    print(image_label_ds)

    BATCH_SIZE = 32

    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    ds = image_label_ds.shuffle(buffer_size=len(paths))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = image_label_ds.apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(labels)))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    print(ds)

    # The dataset may take a few seconds to start, as it fills its shuffle buffer.
    image_batch, label_batch = next(iter(ds))

    return image_batch, label_batch




train_og_paths = all_image_paths[:50]
train_og_labels = all_image_labels[:50]
test_og_paths = all_image_paths[50:]
test_og_labels = all_image_labels[50:]
train_wm_paths = all_watermarked_paths[:50]
train_wm_labels = all_image_labels[:50]
test_wm_paths = all_watermarked_paths[50:]
test_wm_labels = all_image_labels[50:]


train_og_image_batch, train_og_label_batch = build_from_list(
    train_og_paths, train_og_labels)

test_og_image_batch, test_og_label_batch = build_from_list(
    test_og_paths, test_og_labels)

train_wm_image_batch, train_wm_label_batch = build_from_list(
    train_wm_paths, train_wm_labels)

test_wm_image_batch, test_wm_label_batch = build_from_list(
    test_wm_paths, test_wm_labels)

#################################################################################
################### END OF FILE #################################################
#################################################################################
# %%we need to scale the 0-255 values down to 0-1 before feeding them to the model
train_og_image_batch = train_og_image_batch / 255.0

test_og_image_batch = test_og_image_batch / 255.0

train_wm_image_batch = train_wm_image_batch / 255.0

test_wm_image_batch = test_wm_image_batch / 255.0

# %%Build the model -- setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 192, 192, 32)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# %%Build the model -- compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# %%Train the Model
model.fit(train_og_image_batch, train_og_label_batch, epochs=5)


#######################################
########## TEST MODEL #################
#######################################



# %%Evaluate Accuracy
test_loss, test_acc = model.evaluate(test_og_image_batch, test_og_label_batch)
