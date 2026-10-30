# Embedded Digit Recognition on Raspberry Pi

**From PyTorch training to handwritten C++ inference on a live video stream.**

An embedded computer vision project comparing a multilayer perceptron (MLP) and a
small convolutional neural network (CNN) for handwritten digit recognition.
Models are trained in Python with PyTorch; their forward passes are implemented
directly in C++, without a deep learning runtime on the inference path.

The application targets Raspberry Pi 5 and combines OpenCV preprocessing,
digit classification and GStreamer video transport.

## Engineering focus

- **Model implementation:** dense layers, convolutions and ReLU written in C++.
- **Embedded integration:** trained parameters stored as a C++ header for the MLP
  and binary weight files for the CNN.
- **Image processing:** grayscale conversion, thresholding, contour selection,
  square centering, resizing to 28 × 28 and normalization.
- **Video pipeline:** receive an H.264 stream, overlay the predicted digit and
  return the annotated video over TCP.
- **Model comparison:** training notebooks, confusion matrices and a project report.

## Start exploring

| Entry | What to review |
|---|---|
| [Project report](CR_IA_embarque.pdf) | Project setup, experiments and discussion |
| [Model comparison notebook](training/compare_models_.ipynb) | Compare the two model families |
| [CNN training notebook](training/train_cnn.ipynb) | Training and evaluation workflow |
| [MLP training notebook](training/train_mlp_py.ipynb) | Fully connected model workflow |
| [CNN inference](inference_c/CNN/neural_net_cnn.cpp) | Convolution, activation, dense layer and weight loading |
| [MLP inference](inference_c/MLP/neural_network.cpp) | Dense layers and prediction helpers |
| [Video application](inference_c/MLP/main.cpp) | Preprocessing, streaming and prediction overlay |

## Models

| Model | Architecture | Embedded parameters |
|---|---|---|
| MLP | 784 → 128 → 64 → 10, with ReLU between hidden layers | `inference_c/MLP/model_weights.h` |
| CNN | Two 5 × 5 single-channel convolutions with ReLU, then a 784 → 10 linear layer | `inference_c/CNN/*.bin` |

Training uses MNIST and a custom digit dataset. The Python scripts contain
accuracy evaluation and confusion-matrix plots; the C++ applications perform
the video inference. The camera pipeline processes the largest eligible contour,
so it is intended for a single prominent digit.

## Build the C++ applications

Requirements: a C++17 compiler, Make, pkg-config and OpenCV 4 built with GStreamer
support. The video pipelines also require GStreamer elements for H.264 decoding,
x264 encoding and MPEG-TS output.

On Debian or Raspberry Pi OS, the development dependencies can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config libopencv-dev gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

Clone and build:

```bash
git clone https://github.com/amersaidi2002-byte/projet_mnist.git
cd projet_mnist
make -C inference_c/MLP
make -C inference_c/CNN
```

Run either application from its own directory. The CNN reads its binary weights
relative to the current working directory.

```bash
cd inference_c/MLP
./app 5000 8554 1280 720
```

The positional arguments are input port, output port, frame width and frame height.
Defaults are `5000 8554 1280 720`.

The application expects an H.264 TCP source at `127.0.0.1:5000`; it does not open
a camera directly. It serves the annotated MPEG-TS/H.264 stream on port `8554`.
Camera capture must be configured separately. Match the width and height to the
incoming video, and use a trusted local network for the video output.

## Repository map

| Path | Contents |
|---|---|
| `training/` | PyTorch scripts, notebooks and saved model weights |
| `inference_c/MLP/` | C++ MLP, embedded weights and video application |
| `inference_c/CNN/` | C++ CNN, binary weights and video application |
| `Docker/` | Dockerfiles for the inference applications |
| `digits_IA_embarque.zip` | Custom handwritten digit dataset archive |
| `CR_IA_embarque.pdf` | Original project report |

## Scope

This is an academic embedded AI prototype. The notebooks and report document
the experiments; this README does not claim a newly reproduced accuracy or
end-to-end frame rate. Camera behavior depends on illumination, thresholding,
digit placement and the preprocessing used during training.

<details>
<summary>Présentation en français</summary>

Projet d'IA embarquée pour la reconnaissance de chiffres manuscrits sur
Raspberry Pi. Deux modèles, MLP et CNN, sont entraînés avec PyTorch, puis leur
inférence est implémentée directement en C++. OpenCV assure le prétraitement
et GStreamer le transport du flux vidéo annoté. Le projet met en pratique
l'intégration d'un modèle de vision dans une application embarquée.

</details>
