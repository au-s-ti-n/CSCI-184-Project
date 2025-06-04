import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import pillow_heif
from PIL import Image


pillow_heif.register_heif_opener()

INPUT_DIR = "raw_images"
OUTPUT_DIR = "images"
TARGET_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_to_square(img):
  img = img_to_array(img)
  height, width, _ = img.shape
  side = min(height, width)
  cropped = tf.image.resize_with_crop_or_pad(img, target_height=side, target_width=side)
  return cropped

def pad_to_square(img):
  img = img_to_array(img)
  height, width, _ = img.shape
  max_dim = max(width, height)
  padded_img = tf.image.resize_with_pad(img, target_height=max_dim, target_width=max_dim)
  return padded_img

def resize_and_save(image_path, output_path):
  img = Image.open(image_path)
  # img_padded = pad_to_square(img)
  new_img = crop_to_square(img)
  img_resized = tf.image.resize(new_img, TARGET_SIZE)
  img_resized = tf.cast(img_resized, tf.uint8)
  img_pil = array_to_img(img_resized)
  img_pil.save(output_path)


folders = [name for name in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, name))]
print('FOLDERS:', folders)

for folder in folders:
  print(folder)
  os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

  num = 0
  for filename in os.listdir(os.path.join(INPUT_DIR, folder)):
    input_path = os.path.join(INPUT_DIR, folder, filename)
    output_path = os.path.join(OUTPUT_DIR, folder, f'{folder}{num}.jpg')

    if os.path.exists(output_path):
      print('\t skipping', output_path, '(already exists)')
    else:
      print('\t', input_path, '->', output_path)
      resize_and_save(input_path, output_path)
    
    num += 1
