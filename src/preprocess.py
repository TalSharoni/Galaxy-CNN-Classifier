import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
import pickle
from collections import Counter

top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(top_dir, 'clean_data')
output_dir = os.path.join(top_dir, 'preprocessed_data')
images_dir = os.path.join(data_dir, 'images')

os.makedirs(output_dir, exist_ok=True)

metadata_path = os.path.join(data_dir, 'galaxies.csv')
df = pd.read_csv(metadata_path)

df['picture_path'] = df['picture_path'].apply(lambda x: os.path.join(images_dir, os.path.basename(x)))
paths = df['picture_path'].values
label_encoder = LabelEncoder()
df['classification'] = label_encoder.fit_transform(df['classification'])
labels = df['classification'].values

def preprocess_images(paths, target_size=(224, 224)):
    images = []
    for path in paths:
        img = load_img(path)
        img = img_to_array(img) / 255.0

        height, width, _ = img.shape
        crop_height, crop_width = target_size

        top = max(0, (height - crop_height) // 2)
        bottom = top + crop_height
        left = max(0, (width - crop_width) // 2)
        right = left + crop_width
        img = img[top:bottom, left:right]
        img = tf.image.resize(img, target_size).numpy()

        images.append(img)

    return np.array(images)

all_images = preprocess_images(paths)
all_labels = labels

def save_data(data, filename):
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(data, f)

save_data((all_images, all_labels), 'all_data.pkl')

print("Distribution across the dataset:", Counter(all_labels))