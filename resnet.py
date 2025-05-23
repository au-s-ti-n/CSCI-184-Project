import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


root_dir = "images" 
img_size = (224, 224)
batch_size = 32
epochs = 10

image_paths = []
labels = []

for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.png')): # accepts jpg or png
                image_paths.append(os.path.join(class_folder, fname)) # add relative path to root
                labels.append(class_folder)

df = pd.DataFrame({
    'filename': image_paths,
    'class': labels
})

# train test split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)

# image generators
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

# load ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

### -------------- 
### save the model 
### --------------

# predict 
def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = np.argmax(preds)
    class_names = list(train_data.class_indices.keys())
    return class_names[class_idx]

# print(predict_image("images/Kenna/image_name.jpg"))
