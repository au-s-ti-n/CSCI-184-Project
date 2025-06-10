import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

root_dir = "/content/drive/MyDrive/images"  # <-- Update if your path is different
img_size = (224, 224)
batch_size = 32
epochs = 10

image_paths = []
labels = []

for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_folder, fname))  # relative path
                labels.append(class_folder)

df = pd.DataFrame({
    'filename': image_paths,
    'class': labels
})

counts = df['class'].value_counts()
valid_classes = counts[counts >= 2].index.tolist()
df = df[df['class'].isin(valid_classes)].reset_index(drop=True)