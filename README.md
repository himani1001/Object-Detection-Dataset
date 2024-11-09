# Custom Aquarium Object Detection with YOLOv8

This project involves training a custom object detection model using YOLOv8 on an aquarium dataset. The model is designed to detect various objects within aquarium images, and it is trained with custom annotations and dataset-specific configurations.

## Dataset: https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots

## Project Structure

- **Data Preparation**: The dataset is extracted from a ZIP archive and is assumed to contain images and annotation files in the correct format for YOLOv8 training.
- **Model Training**: The YOLOv8 model is trained with custom data, including image resizing, batch processing, and running on a CUDA-enabled GPU for faster training.
- **Validation & Inference**: After training, the model is evaluated using validation metrics and tested with sample images for object detection predictions.

## Key Features

- **YOLOv8 Model**: The project utilizes the YOLOv8 architecture for efficient object detection, which is well-suited for real-time applications.
- **Custom Dataset**: A custom dataset containing aquarium images with pre-labeled annotations is used for training.
- **Training Hyperparameters**:
  - **Epochs**: The model is trained for 5 epochs (can be adjusted based on dataset size and convergence).
  - **Batch Size**: The batch size is set to 4 for optimal GPU memory usage.
  - **Image Size**: Input images are resized to 640x640 for training.
  - **CUDA**: Training is done on a GPU to accelerate the process.

## Steps to Run

1. **Extract the Dataset**:
   The dataset is extracted from a ZIP file located at `/content/archive.zip` and placed in `/content`.

2. **Install Dependencies**:
   The project uses the `ultralytics` package, which is installed using the following command:
   ```bash
   pip install ultralytics
   ```

3. **Train the Model**:
   The model is trained using the YOLOv8 framework. The training parameters (epochs, batch size, etc.) can be adjusted as needed.

4. **Evaluate the Model**:
   After training, the model is evaluated on a validation dataset, and key metrics like Precision, Recall, mAP, and IoU are analyzed.

5. **Make Predictions**:
   After the model is trained, predictions are made on the test dataset, and the results are saved.

## Metrics

The model's performance is evaluated using the following metrics:

- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to find all positive instances.
- **mAP (mean Average Precision)**: Overall accuracy measurement averaged over different recall levels.
- **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth bounding boxes.

Example results (based on my dataset):

- Precision: 0.85
- Recall: 0.80
- mAP: 0.75
- IoU: 0.82

## Notes on Augmentation

The project does not explicitly include data augmentations in the code, as the YOLOv8 model might already incorporate augmentation strategies. However, common augmentations that could be applied (if needed) include:

- **Horizontal Flipping**: To simulate different perspectives.
- **Scaling/Zooming**: To handle objects of varying sizes.
- **Rotation**: To make the model invariant to rotations.

## Requirements

- **Python 3.x**
- **PyTorch** with CUDA support (for GPU training)
- **ultralytics**: YOLOv8 library
- **Other dependencies**: `torch`, `os`, `zipfile`

## Conclusion

This project demonstrates how to fine-tune a YOLOv8 model on a custom dataset for object detection tasks. It provides a clear and efficient approach to training and evaluating a real-time object detection model, specifically for aquarium-related objects.
