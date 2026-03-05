import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from ultralytics import YOLO

# Set up argument parser (to get inputs directly from CLI)
parser = argparse.ArgumentParser(description="Image Style Transfer Script")

# Add arguments for the image filenames and operation mode
parser.add_argument("content_image", type=str, help="Filename of the content image in the 'inputs' folder")
parser.add_argument("style_image", type=str, help="Filename of the style image in the 'inputs' folder")
parser.add_argument("mode", type=str, choices=["full","person", "fg", "lower","upper"], 
                    help="Mode of style transfer: 'full', 'fg','person','upper' or 'lower'")
# Add NUM_ITER as an optional argument with a default value
parser.add_argument("--num_iter", type=int, default=300, 
                    help="Number of iterations for style transfer (default: 300)")
# Add SAVE_SEG as a boolean flag with a default value of False
parser.add_argument("--save_seg", action='store_true', 
                    help="Save segmentation results (default: False)")

args = parser.parse_args() # Parse the arguments

# Constants
RESIZE_HEIGHT = 600
NUM_ITER = args.num_iter
SAVE_SEG = args.save_seg
CONTENT_WEIGHT = 8e-4
STYLE_WEIGHT = 8e-1

# The layer to be used for the content loss.
CONTENT_LAYER_NAME = "block5_conv2"

# List of layers to be used for the style loss.
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

def combine(img1, img2, binary_mask):
    print("Combining Images...")
    binary_mask = np.array(binary_mask, dtype=np.float32)
    mask_resized = cv2.resize(binary_mask, (img1.shape[1], img1.shape[0]))
    
    mask_expanded = np.expand_dims(mask_resized, axis=-1) 
    mask_3channel = np.repeat(mask_expanded, 3, axis=-1)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    combined_image = mask_3channel * img2 + (1 - mask_3channel) * img1
    combined_image = combined_image.astype(np.uint8)
    return combined_image

def get_result_image_size(np_image, result_height):
    image_height, image_width = np_image.shape[:2]
    result_width = int(image_width * result_height / image_height)
    return result_height, result_width

def preprocess_image(np_image, target_height, target_width):
    img = tf.image.resize(np_image, (target_height, target_width)).numpy()
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

def get_model():
    model = vgg19.VGG19(weights = 'imagenet', include_top = False)
    # Set all layers to be non-trainable
    for layer in model.layers:
        layer.trainable = False
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    return keras.Model(inputs = model.inputs, outputs = outputs_dict)

def get_optimizer():
    return keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 8.0, decay_steps = 445, decay_rate = 0.98
        )
    )

def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    loss_content = compute_content_loss(content_features, combination_features)
    loss_style = compute_style_loss(style_features, combination_features, combination_image.shape[1] * combination_image.shape[2])

    return CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

# A loss function designed to maintain the 'content' of the original image in the generated_image
def compute_content_loss(content_features, combination_features):
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]

    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2

# loss function designed to maintain the 'style' features of the style image in the generated_image
def compute_style_loss(style_features, combination_features, combination_size):
    loss_style = 0

    for layer_name in STYLE_LAYER_NAMES:
        style_feature = style_features[layer_name][0]
        combination_feature = combination_features[layer_name][0]
        loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)

    return loss_style

def style_loss(style_features, combination_features, combination_size):
    S = gram_matrix(style_features)
    C = gram_matrix(combination_features)
    channels = style_features.shape[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

def gram_matrix(x):
   x = tf.transpose(x, (2, 0, 1))
   features = tf.reshape(x, (tf.shape(x)[0], -1))
   gram = tf.matmul(features, tf.transpose(features))
   return gram

# Function to convert a tensor into a valid image
def deprocess_image(tensor, result_height, result_width):
    tensor = tensor.numpy()
    tensor = tensor.reshape((result_height, result_width, 3))
    # Removing zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680

    tensor = tensor[:, :, ::-1] # 'BGR'->'RGB'
    return np.clip(tensor, 0, 255).astype("uint8")

def style_transfer(img, style_img):
  result_height, result_width = get_result_image_size(img, RESIZE_HEIGHT)

  content_tensor = preprocess_image(img, result_height, result_width)
  style_tensor = preprocess_image(style_img, result_height, result_width)
  generated_image = tf.Variable(preprocess_image(img, result_height, result_width))

  # Building model
  model = get_model()
  optimizer = get_optimizer()

  content_features = model(content_tensor)
  style_features = model(style_tensor)

  # Optimizing result image
  print("Transfering Style...")
  for iter in range(NUM_ITER):
      if iter % 10 == 0:
          print(f"Iteration: {iter+1}-{min(iter+10,NUM_ITER)}/{NUM_ITER}")
      with tf.GradientTape() as tape:
          loss = compute_loss(model, generated_image, content_features, style_features)
      grads = tape.gradient(loss, generated_image)
      optimizer.apply_gradients([(grads, generated_image)])

  generated_image = deprocess_image(generated_image, result_height, result_width)
  return generated_image


# MAIN SCRIPT
inputs_folder = "inputs"

# Construct the full paths to the images in the inputs folder
content_image_path = os.path.join(inputs_folder, args.content_image)
style_image_path = os.path.join(inputs_folder, args.style_image)
user_want = args.mode

image = keras.preprocessing.image.load_img(content_image_path)
image = keras.preprocessing.image.img_to_array(image)
image_copy = image.copy()
style_img = keras.preprocessing.image.load_img(style_image_path)
style_img = keras.preprocessing.image.img_to_array(style_img)

if user_want == "full":
  # Apply style transfer to the whole image and save
  keras.preprocessing.image.save_img("outputs/" + args.content_image[:-4] + "_result.jpg", style_transfer(image, style_img))

elif user_want == "fg":
  print("Segmenting Image...")
  model = YOLO("seg_models/yolo11l-seg.pt")
  results = model(content_image_path, save=SAVE_SEG)

  masks = []
  for i, mask in enumerate(results[0].masks.data):
    binary_mask = (mask > 0.5).float()  # Threshold mask to binary
    masks.append(binary_mask.cpu().numpy())

  fg = np.zeros_like(masks[0])
  for i, val in enumerate(results[0].boxes.data):
    fg = np.logical_or(fg, masks[i]).astype(int)  

  # Apply style transfer to image, combine with foreground mask, and save
  image = style_transfer(image, style_img)
  image_copy = cv2.resize(image_copy, (image.shape[1], image.shape[0]))
  result_img = combine(image, image_copy, fg)
  keras.preprocessing.image.save_img("outputs/" + args.content_image[:-4] + "_result.jpg", result_img)

elif user_want == "person":
  print("Segmenting Image...")
  model = YOLO("seg_models/yolo11l-seg.pt")
  results = model(content_image_path, save=SAVE_SEG)
  masks = []
  for i, mask in enumerate(results[0].masks.data):
    binary_mask = (mask > 0.5).float()  
    masks.append(binary_mask.cpu().numpy())

  person = np.zeros_like(masks[0])
  for i, val in enumerate(results[0].boxes.data):
    if int(val[-1]) == 0: 
      person = np.logical_or(person, masks[i]).astype(int)

  # Apply style transfer, combine with person mask, and save
  image = style_transfer(image, style_img)
  image_copy = cv2.resize(image_copy, (image.shape[1], image.shape[0]))
  result_img = combine(image, image_copy, person)
  keras.preprocessing.image.save_img("outputs/" + args.content_image[:-4] + "_result.jpg", result_img)

elif user_want == "lower" or user_want == "upper":
    print("Segmenting Image...")
    model = YOLO("seg_models/yolov11l_dress_segment_model.pt")
    results = model(content_image_path, save=SAVE_SEG)
    masks = []
    for i, mask in enumerate(results[0].masks.data):
        binary_mask = (mask > 0.5).float()
        masks.append(binary_mask.cpu().numpy())

    lower = np.zeros_like(masks[0])
    upper = np.zeros_like(masks[0])
    lower_box = []
    upper_box = []

    print("Extracting Dress Bounding Boxes...")
    for i, val in enumerate(results[0].boxes.data):
      if int(val[-1]) in [0, 1, 2, 4, 7]:  
        upper = np.logical_or(upper, masks[i]).astype(int)
        # Track bounding box for upper dress
        if len(upper_box) == 0:
            upper_box = val[:4].cpu().numpy()
        else:
            upper_box[0] = min(upper_box[0], val[0].cpu().numpy())
            upper_box[1] = min(upper_box[1], val[1].cpu().numpy())
            upper_box[2] = max(upper_box[2], val[2].cpu().numpy())
            upper_box[3] = max(upper_box[3], val[3].cpu().numpy())
      else: 
        lower = np.logical_or(lower, masks[i]).astype(int)
        # Track bounding box for lower dress
        if len(lower_box) == 0:
            lower_box = val[:4].cpu().numpy()
        else:
            lower_box[0] = min(lower_box[0], val[0].cpu().numpy())
            lower_box[1] = min(lower_box[1], val[1].cpu().numpy())
            lower_box[2] = max(lower_box[2], val[2].cpu().numpy())
            lower_box[3] = max(lower_box[3], val[3].cpu().numpy())

    lower_box = lower_box.astype(int)
    upper_box = upper_box.astype(int)

    if user_want == "lower":
      # Extract lower dress bounding box, apply style transfer, and resize
      img = image[lower_box[1]:lower_box[3], lower_box[0]:lower_box[2], :]
      width = lower_box[2] - lower_box[0]
      height = lower_box[3] - lower_box[1]
      img = style_transfer(img, style_img)
      img = cv2.resize(img, (width, height))
      image[lower_box[1]:lower_box[3], lower_box[0]:lower_box[2], :] = img

      # Combine and save
      result_img = combine(image_copy, image, lower)
      keras.preprocessing.image.save_img("outputs/" + args.content_image[:-4] + "_result.jpg", result_img)

    elif user_want == "upper":
      # Extract upper dress bounding box, apply style transfer, and resize
      img = image[upper_box[1]:upper_box[3], upper_box[0]:upper_box[2], :]
      width = upper_box[2] - upper_box[0]
      height = upper_box[3] - upper_box[1]
      img = style_transfer(img, style_img)
      img = cv2.resize(img, (width, height))
      image[upper_box[1]:upper_box[3], upper_box[0]:upper_box[2], :] = img

      # Combine and save
      result_img = combine(image_copy, image, upper)
      keras.preprocessing.image.save_img("outputs/" + args.content_image[:-4] + "_result.jpg", result_img)
