# Masked Image Style Transfer using Neural Networks and YOLO

> Apply artistic styles to specific parts of an image using VGG19 for style transfer and YOLO for segmentation-based masking.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modes](#modes)
- [Examples](#examples)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [References](#references)

---

## Overview

This project applies **neural style transfer** to images with the added ability to target specific regions — such as the background, a person, or clothing — rather than styling the whole image at once. It combines **YOLO segmentation** to identify and isolate regions of interest with a **VGG19-based neural network** that performs the actual style transfer. The result is a system that gives precise control over where a particular artistic style is applied.

## Features

- **Full Image Style Transfer**: Apply the style to the entire image without any masking.
- **Selective Transfer**: Apply styles only to the background, the person, or specific clothing regions.
- **Multiple Modes**: Five modes are available — `full`, `person`, `fg`, `upper`, `lower`.
- **Optimized Performance**: Uses VGG19 for feature extraction and YOLO for segmentation, with separate loss functions for content and style.

## Installation

To set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/siddharthjaiswal05/style_transfer_masking.git
   cd style_transfer_masking
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Make sure your environment supports TensorFlow, YOLO (Ultralytics), OpenCV, and other listed packages.

## Usage

Run the style transfer using the following command:

```bash
python style_transfer.py <content_image> <style_image> <mode> [--num_iter <iterations>] [--save_seg]
```

### Arguments

- **content_image**: File name of the content image placed inside the `inputs/` folder.
- **style_image**: File name of the style image placed inside the `inputs/` folder.
- **mode**: The masking mode. Choose from: `full`, `person`, `fg`, `upper`, `lower`.
- **--num_iter** *(optional)*: Number of optimization iterations (default: 300).
- **--save_seg** *(optional)*: If set, the segmentation mask will be saved to disk.

Example:
```bash
python style_transfer.py image.jpg style.jpg person --num_iter 1000 --save_seg
```

## Modes

Five modes are available to control which part of the image receives the style:

- **full**: Style is applied to the entire image with no masking.
- **person**: Style is applied to everything except the detected person(s).
- **fg (foreground)**: Style is applied to the background only, leaving the foreground untouched.
- **upper**: Style is applied only to the upper-body clothing of detected persons.
- **lower**: Style is applied only to the lower-body clothing of detected persons.

Each mode uses a binary mask produced by the segmentation model to determine which pixels are modified.

## Examples

### Full Image Style Transfer ("full" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a7b51773-6d44-49c0-94a2-aea7ede58328" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/aafbd691-dc36-4f2d-9c4d-ca0b5b03d7ec" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/e7851abc-b0b4-443c-b466-ea5125b8bdbe" width="250" height="250"></td>
  </tr>
</table>

### Clothing Style Transfer ("upper" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ed4278e1-8ed1-44dc-a8cf-7c0f1cb23364" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/5ff49770-6d4f-476c-9949-d191f9788bc2" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/ae8de79b-1e05-4e26-ae72-d7f17fd2e1d3" width="250" height="250"></td>
  </tr>
</table>

### Person-Background Style Transfer ("person" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/56e64cd9-49ed-404b-b00e-fe934602137c" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/3583cda2-184f-44a3-954f-ae81b34d19fb" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/330a7efc-d5cf-42b8-a917-354a9fb3ac4a" width="250" height="250"></td>
  </tr>
</table>

### Background Style Transfer ("fg" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0ce7dd80-be26-4493-a476-2dee0c40d430" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/5ff49770-6d4f-476c-9949-d191f9788bc2" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/b8ef8366-4c85-4a6e-ae90-6d8148f4995d" width="250" height="250"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/37e11da5-5075-432c-9b7b-422f9575c603" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/26fe9ef2-5450-430e-b430-d7ddbc2e6e00" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/0e1834c4-a3d2-40c8-a61b-72e00dbe86d5" width="250" height="250"></td>
  </tr>
</table>

---

## Model Architecture

This section explains the two main components of the system — the VGG19-based style transfer network and the YOLO segmentation model — and how they work together.

### 1. VGG19 for Style Transfer

VGG19 is a convolutional neural network originally trained on the ImageNet dataset for image classification. It has 19 layers — 16 convolutional layers and 3 fully connected layers. In this project, it is not used for classification. Instead, it is used as a fixed feature extractor, meaning its weights are not updated during the style transfer process.

**Why VGG19?**  
Convolutional layers in a deep network learn to detect increasingly abstract features. Early layers respond to low-level patterns like edges and colors, while deeper layers capture high-level structures like shapes and objects. This property makes VGG19 useful for separating "content" (what is in the image) from "style" (how it looks in terms of texture and color patterns).

**Content Representation:**  
The content of an image is captured using the activations from a deep convolutional layer, specifically `block4_conv2` in VGG19. At this depth, the network has encoded the structural layout of the image — positions of objects, overall composition — without holding onto fine pixel-level details. The content loss is the mean squared difference between the feature maps of the generated image and the content image at this layer.

**Style Representation:**  
Style is represented differently. Rather than using activations directly, the style is captured using the **Gram matrix** of the feature maps. The Gram matrix is computed by taking the dot product of the feature map with its own transpose, after reshaping it into a 2D matrix. This operation captures the correlation between different feature channels — essentially encoding which textures and patterns tend to appear together, independent of their spatial location in the image. Style is measured across multiple layers: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, and `block5_conv1`. Using multiple layers ensures both fine-grained textures (from early layers) and broader stylistic patterns (from deeper layers) are captured. The style loss is the mean squared difference between the Gram matrices of the generated image and the style image across all these layers.

**Optimization Process:**  
The generated image starts as a copy of the content image (or random noise) and is updated iteratively using gradient descent. At each step, the total loss — a weighted sum of content loss and style loss — is computed, and the pixel values of the generated image are adjusted to reduce this loss. Over several hundred iterations, the image gradually takes on the visual texture and color patterns of the style image while retaining the structural content of the content image.

**Total Loss:**

```
Total Loss = alpha * Content Loss + beta * Style Loss
```

The weights `alpha` and `beta` control how much content versus style is preserved. A higher `beta/alpha` ratio results in stronger stylization.

---

### 2. YOLO Segmentation for Masking

YOLO (You Only Look Once) is a real-time object detection and segmentation model. In this project, the YOLOv11 segmentation variant is used to generate pixel-level binary masks for different regions of the image.

**How YOLO Segmentation Works:**  
Unlike standard object detection that produces bounding boxes, segmentation models output a mask for each detected object. Each mask is a binary image of the same size as the input, where pixels belonging to the detected region are marked as 1 and all other pixels are 0.

Two separate YOLO models are used in this project:

- **Person segmentation model**: A standard YOLOv11 segmentation model trained to detect and segment people. This is used for the `person` and `fg` modes.
- **Clothing segmentation model**: A YOLOv11 segmentation model fine-tuned on a custom clothing dataset to separately detect upper-body clothing (shirts, jackets, etc.) and lower-body clothing (pants, skirts, etc.). This is used for the `upper` and `lower` modes.

The clothing model required custom training because standard YOLO models are not trained to differentiate between clothing categories at a fine-grained level.

**Mask Generation and Application:**  
Once a mask is generated for the desired region, it is used to blend the styled image and the original image. Pixels inside the masked region retain the original image values, while pixels outside receive the styled output. This gives a clean boundary between styled and unstyled areas.

```
output_pixel = mask * original_pixel + (1 - mask) * styled_pixel
```

This simple but effective combination step is what enables selective style transfer without visible artifacts at region boundaries.

---

## Implementation Details

### Directory Structure

```
style_transfer_masking/
├── inputs/          # Place content and style images here
├── outputs/         # Styled output images are saved here
├── style_transfer.py
├── requirements.txt
└── README.md
```

### Pipeline Summary

1. Load the content image and style image from the `inputs/` folder.
2. If the mode is not `full`, run the appropriate YOLO model to generate a binary mask for the target region.
3. Pass the content and style images through VGG19 to extract feature maps and compute initial content and style targets.
4. Initialize the generated image and run the optimization loop for the specified number of iterations, minimizing the total loss.
5. After optimization, apply the binary mask to blend the styled result with the original image.
6. Save the final output to the `outputs/` folder. Optionally save the segmentation mask if `--save_seg` is set.

---

## References

- Gatys et al., [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), CVPR 2016.
- [Style Transfer Reference Implementation](https://github.com/superb20/Image-Style-Transfer-Using-Convolutional-Neural-Networks?tab=readme-ov-file) — used as reference for coding the style transfer algorithm.
- [Ultralytics YOLOv11 Documentation](https://github.com/ultralytics/ultralytics) — used for training the custom clothing segmentation model.

---
