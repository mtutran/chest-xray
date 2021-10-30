import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import densenet
import albumentations as augment
from static_values.values import IMAGE_SIZE, BATCH_SIZE

autotune = tf.data.AUTOTUNE


def classify_augmentation(training=False):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=90, quality_upper=100, p=0.4),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.025, rotate_limit=45),
            augment.GaussNoise(p=0.4),
            augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ])
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)])

    def preprocess_image(image_file):
        image_raw = tf.io.read_file(image_file)
        decoded = tf.image.decode_jpeg(image_raw, channels=3)
        data = {'image': decoded.numpy()}
        aug_img = transform(**data)['image']
        aug_img = tf.cast(aug_img, tf.float32)
        tensor = densenet.preprocess_input(aug_img)
        return tensor

    return preprocess_image


def ClassifyGenerator(images, y, image_dir, training=False, batch_size=BATCH_SIZE):
    def process_data(image_file, label):
        aug_img = tf.numpy_function(func=classify_augmentation(training), inp=[image_file], Tout=tf.float32)
        return aug_img, label

    images_ts = tf.data.Dataset.from_tensor_slices(image_dir + images)
    labels_ts = []
    for col in range(y.shape[1]):
        labels_ts.append(tf.data.Dataset.from_tensor_slices(y[:, col].astype(float)))
    labels_ts = tf.data.Dataset.zip(tuple(labels_ts))
    ds = tf.data.Dataset.zip((images_ts, labels_ts))
    ds = ds.shuffle(24 * batch_size, reshuffle_each_iteration=training)
    ds = ds.map(lambda x, y: process_data(x, y),
                num_parallel_calls=autotune).batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds
