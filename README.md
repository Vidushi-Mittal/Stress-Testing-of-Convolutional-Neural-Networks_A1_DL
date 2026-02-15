# Stress-Testing-of-Convolutional-Neural-Networks_A1_DL

# Deep Learning Assignment: Fashion-MNIST Classification
##**********************************************************************PART 1**********************************************
## Model Selection overview
The first part of the project implements and evaluates various Convolutional Neural Network (CNN) architectures for image classification on the Fashion-MNIST dataset. The primary goal is to compare the performance of pre-trained models (ResNet-18, VGG-16, VGG-19) adapted for Fashion-MNIST with a custom-designed baseline CNN, and to investigate the impact of data augmentation techniques.

## Implementation Team
- M25MAC004
- M25MAC008
- M25MAC014
- M25MAC016

## Dataset: Fashion-MNIST
The Fashion-MNIST dataset is a collection of 28x28 grayscale images of 10 fashion categories, often used as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It consists of 60,000 training images and 10,000 test images.

**Dataset Details:**
- **Image Dimensions:** (1, 28, 28) (Channel, Height, Width)
- **Number of Classes:** 10
- **Classes:** ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

**Data Preprocessing:**
- Images are transformed to PyTorch tensors and normalized with a mean of 0.2860 and standard deviation of 0.3530, derived from the Fashion-MNIST dataset itself.

**Data Split:**
- **Training Set:** 50,000 samples
- **Validation Set:** 10,000 samples
- **Test Set:** 10,000 samples
- **Batch Size:** 128

## Models Implemented

### 1. Adapted Pre-trained Models (ResNet-18, VGG-16, VGG-19)
To adapt these models, originally designed for ImageNet (3-channel, 224x224 images), to Fashion-MNIST (1-channel, 28x28 images), the following modifications were applied via the `train_prep_img` function:
- The first convolutional layer (`conv1` for ResNet, `features[0]` for VGG) was modified to accept 1 input channel.
- Max-pooling layers were either removed (ResNet) or strategically modified/replaced with `nn.Identity()` (VGG) to prevent premature dimension collapse due to the small input image size (28x28).
- The final fully connected layer was replaced with a new `nn.Linear` layer to output 10 classes.
- All model weights were initialized from scratch (i.e., `weights=None`).

### 2. Custom Baseline CNN
A custom-designed CNN architecture featuring two convolutional blocks followed by a classifier head. Each block consists of `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d` -> `Dropout2d`. The classifier uses `nn.Flatten` and two `nn.Linear` layers with `BatchNorm1d` and `Dropout`.

## Training Pipeline (`run_training_pipeline`)

All models are trained using a standardized pipeline:
- **Loss Function:** `nn.CrossEntropyLoss`
- **Optimizer:** `optim.SGD` with `lr=0.01`, `momentum=0.9`, `weight_decay=5e-4`
- **Learning Rate Scheduler:** `optim.lr_scheduler.CosineAnnealingLR` with `T_max=50`
- **Epochs:** 15 epochs for initial runs, 30 epochs for the best performing augmented model.
- **Device:** Training on GPU (`cuda`) if available, otherwise CPU.
- Best model weights (based on validation accuracy) are saved during training.

## Data Augmentation Experiments (Custom CNN)

### 1. Simple Data Augmentation
- `transforms.RandomHorizontalFlip(p=0.5)`
- `transforms.RandomRotation(degrees=10)`

### 2. Improved Data Augmentation (Crop + Erasing)
- `transforms.RandomCrop(28, padding=2)`
- `transforms.RandomHorizontalFlip(p=0.5)`
- `transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))`

## Results Summary

| Model                                  | Final Test Accuracy (15 Epochs) | Final Test Accuracy (30 Epochs, for best augmented model) |
|----------------------------------------|---------------------------------|-----------------------------------------------------------|
| VGG16 (Adapted)                        | 92.03%                          | -                                                         |
| VGG19 (Adapted)                        | 91.78%                          | -                                                         |
| ResNet-18 (Adapted)                    | 90.38%                          | -                                                         |
| Custom Baseline CNN (No Augmentation)  | 92.97%                          | -                                                         |
| Custom CNN + Simple Augmentation       | 92.43%                          | 93.24%                                                    |
| Custom CNN + Improved Augmentation     | 91.97%                          | -                                                         |

### Key Observations:
- The custom baseline CNN performed competitively, even outperforming the adapted ResNet-18 and VGG models initially.
- Simple data augmentation slightly improved the performance of the Custom CNN when trained for 30 epochs.
- Surprisingly, the 'Improved Augmentation' (Crop + Erasing) did not yield better results than simple augmentation on this dataset, indicating that excessive or inappropriate augmentation can sometimes hinder performance or require more fine-tuning.

## Visualizations
Training and validation loss/accuracy curves are generated for each model, along with the final test accuracy, and saved as PNG files (e.g., `vgg16_training_curves.png`, `custom_baseline_cnn_training_curves.png`).

## Reproducibility
- A fixed seed (`56`) is used for `torch`, `torch.cuda`, `numpy`, and `random` to ensure reproducibility of results.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are set.

## Usage
To run this notebook:
1. Ensure you have a Colab environment or a local Python environment with PyTorch and `torchvision` installed.
2. Execute the cells sequentially. The dataset will be automatically downloaded.
3. The training pipelines for each model will run, and performance metrics will be printed, along with saved training curves.
