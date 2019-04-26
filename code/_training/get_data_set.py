# adapted from https://www.tensorflow.org/alpha/tutorials/load_data/images#setup

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pathlib
import random
import IPython.display as display
import matplotlib.pyplot as plt

tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Retrieve the images


#the following line will only work if you already have the stuff downloaded.
#data_root_orig =  "C:\\Users\\sheetzeg\\.keras\\datasets\\flower_photos"
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)

# list the directories retrieved
for item in data_root.iterdir():
  print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("Image count: ", image_count)


#the following code could be used to have a look at the images.
import os
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    # added as_posix() to this line, because windows path doesn't match LICENSE.txt format.
    image_rel = pathlib.Path(image_path).relative_to(data_root).as_posix()
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])
#
# for n in range(3):
#   image_path = random.choice(all_image_paths)
#   display.display(display.Image(image_path))
#   print(caption_image(image_path))
#   print()


# determine the label for each image
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print("\nThe label names as pulled from the directory names are: ", label_names)

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print("The label_to_index dict is: ", label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("We made a list of all the images labels. First 10 labels indices: ", all_image_labels[:10])

#######################################
######## Convert from JPEG to Tensor####
########################################
def get_tensor_print_jpeg(path, label):
    img_tensor = load_and_preprocess_image(path)
    plt.imshow(img_tensor)
    plt.grid(False)
    plt.xlabel(caption_image(path))
    plt.title(label_names[label].title())
    plt.show()
    print()
    return img_tensor

def load_and_preprocess_image(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor=preprocess_image(img_raw)
    return img_tensor

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image

# get an image, convert to tensor, print
# img_final = get_tensor_print_jpeg(all_image_paths[78], all_image_labels[78])

# Build a tf.data.Dataset
print("\n\nBuilding A Dataset...")
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

################################
#######prep dataset for training#####
########################


BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

###############################
##pipe dataset to a model########
####################


mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False
def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)
# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))])
logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
len(model.trainable_variables)
print("mdodel summary: ", model.summary())
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
print("steps per epoch: ", steps_per_epoch)
print(model.fit(ds, epochs=1, steps_per_epoch=3))
