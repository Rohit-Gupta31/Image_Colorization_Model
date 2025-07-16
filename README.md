# Image_Colorization_Model
This project implements an **image colorization system** using a deep **Convolutional Autoencoder with skip connections**. Given a grayscale landscape image, the model predicts a fully colorized RGB version. The approach is inspired by U-Net and encoder-decoder architectures.
---

## 🧠 Model Architecture

The model consists of an **encoder-decoder** pipeline with skip connections that pass features from early encoder layers directly into later decoder layers. This helps the model retain spatial and edge details lost during downsampling.

### 🔹 Encoder

- `Conv2D(64, 3, strides=2)` → Downsample
- `Conv2D(128, 3, strides=2)`
- `Conv2D(256, 3, strides=2)`
- `Conv2D(512, 3, strides=2)` → Bottleneck

### 🔹 Decoder

- `Conv2DTranspose(256, 3, strides=2)` → Concatenate with encoder layer 3
- `Conv2DTranspose(128, 3, strides=2)` → Concatenate with encoder layer 2
- `Conv2DTranspose(64, 3, strides=2)` → Concatenate with encoder layer 1
- `Conv2DTranspose(3, 3, strides=2, activation='sigmoid')` → Final RGB output

> Skip connections are implemented using `layers.Concatenate()` to fuse encoder features with decoder inputs at matching spatial resolutions.

---

## 🖼️ Dataset

**Dataset**: [Landscape Color and Grayscale Images](https://www.kaggle.com/datasets/fuadhasan/landscape-image-colorization)

- Contains pairs of grayscale and corresponding RGB landscape images
- Images are resized to `256x256` before training
- The grayscale images serve as model input, and RGB images as ground truth labels

---

## 🚀 How to Use

1. Download and unzip the dataset from Kaggle
2. Open `Image Colorization.ipynb` in Jupyter or Google Colab
3. Follow the notebook:
   - Preprocess images
   - Define the autoencoder model
   - Train the model
   - Visualize predictions

---

## 🧪 Sample Results

The model produces vibrant colorizations of grayscale landscapes, restoring textures and tones convincingly.

| Grayscale Input | Colorized Output |
|------------------|-------------------|
| ![](examples/input.png) | ![](examples/output.png) |

---

## ⚙️ Technical Details

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Output Activation**: `sigmoid` to ensure pixel values stay in [0, 1]
- **Evaluation**: Visual comparison of predicted RGB images against ground truth

---

## 🛠️ Dependencies

Install the required Python packages using:

```bash
pip install tensorflow numpy matplotlib pillow tqdm

