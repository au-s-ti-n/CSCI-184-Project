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

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['class'], random_state=42)

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=root_dir,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

val_data = val_gen.flow_from_dataframe(
    val_df,
    directory=root_dir,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

test_data = val_gen.flow_from_dataframe(
    test_df,
    directory=root_dir,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(len(train_data.class_indices), activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc:.2f}")

