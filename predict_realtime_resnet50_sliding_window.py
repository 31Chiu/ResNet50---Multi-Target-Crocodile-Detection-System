# PyTorch deep learning framework
import torch
# Neural network modules from PyTorch
import torch.nn as nn
# Import ResNet50 model architecture
from torchvision.models import resnet50, ResNet50_Weights
# Python Imaging Library for image processing
from PIL import Image
# Image transformation utilities from torchvision
import torchvision.transforms as transforms
# OpenCV for real-time video capture and processing
import cv2
# NumPy for numerical operation
import numpy as np
# Time for measuring frame performance
import time
# System-level operation
import sys
# Operating system path operations
import os

def load_model(model_path):
    """
    Load and initialize the ResNet50 model from a checkpoint file.
    Args:
        model_path (str): Path to the saved model checkpoint.
    Returns:
        tuple: (model, class_names) - The loaded model and list of class names.
    """
    # Load the saved model checkpoint into CPU memory
    checkpoint = torch.load(model_path, map_location='cpu')

    # Initialize the ResNet50 model architecture without pretrained weights
    # The architecture must match the one used during training.
    model = resnet50(weights=None)

    # Get the number of input features for the final fully connected layer
    num_features = model.fc.in_features
    # Replace the final layer with the custom classifier from your training script
    model.fc = nn.Sequential(
        nn.Dropout(0.5),                                    # Add dropout for regularization
        nn.Linear(num_features, len(checkpoint['classes'])) # New classification layer
    )

    # Load the trained weights from the checkpoint into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode (disable dropout and batch norm effects)
    model.eval()

    return model, checkpoint['classes']

def process_patch(patch):
    """
    Process a video patch for model input.
    Args:
        patch: BGR format patch from OpenCV.
    Returns:
        torch.Tensor: Processed image tensor ready for model input.
    """
    # Convert BGR patch to RGB and create PIL Image object
    img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

    # Define image transformations pipeline (must be identical to validation transform)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transformations and add batch dimention
    return transform(img).unsqueeze(0)

def predict(model, image_tensor, class_names):
    """
    Perform prediction on an input image tensor.
    Args:
        model: The loaded ResNet50 model.
        image_tensor: Preprocessed image tensor.
        class_names: List of class names.
    Returns:
        tuple: (predicted_class, confidence) - The predicted class name and confidence score.
    """
    # Disable gradient calculation for inference to save memory and computations
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image_tensor)
        # Get the index of the highest score
        _, predicted = torch.max(outputs, 1)
        # Convert raw outputs to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Return the predicted class name and its confidence score
    return class_names[predicted[0]], probabilities[0][predicted[0]].item()

# Sliding window generator
def sliding_window(image, step_size, window_size):
    # Slide the window from top to bottom and from left to right
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Yield the window coordinates and the image patch itself
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def main():
    # Main function to run real-time object detection with ResNet50.
    # Path to the trained ResNet50 model checkpoint
    # This path is updated to match your training script's output.
    model_path = 'resnet50_checkpoint/best_resnet50_model.pth'

    if not os.path.exists(model_path):
        print(f'Error: Model file not found at {model_path}')
        print("Please ensure you have trained the ResNet50 model and the path is correct.")
        return
    
    print("Loading ResNet50 model...")
    # Load the model and get class names
    model, class_names = load_model(model_path)

    # Define your target class
    TARGET_CLASS = 'crocodile' # Change this to your desired class name

    if TARGET_CLASS not in class_names:
        print(f"Error: Target class {TARGET_CLASS} not found in the model's class list: {class_names}.")
        return
    print(f'Model loaded successfully! Looking for: {TARGET_CLASS}')

    # Initialize video capture from default camera (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    print("\nPress 'q' to exit.")

    # Define sliding window parameters
    (winW, winH) = (128, 128)   # Window width and height
    stepSize = 32               # The pixel distance of each slide
    CONF_THRESHOLD = 0.90       # Confidence threshold for predictions

    # Main processing loop
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # List to store detections for the current frame
        detections = []

        # Sliding window core logic
        for (x, y, window) in sliding_window(frame, step_size=stepSize, window_size=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Process the window patch for the model
            image_tensor = process_patch(window)
            # Make a prediction
            predicted_class, confidence = predict(model, image_tensor, class_names)

            # If the prediction is our target class and confidence is high enough, save it
            if predicted_class == TARGET_CLASS and confidence >= CONF_THRESHOLD:
                detections.append((x, y, x + winW, y + winH))

        # Draw rectangles around detected objects
        for (startX, startY, endX, endY) in detections:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f'{TARGET_CLASS}: {confidence: .2f}'
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f'FPS: {fps: .2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow("ResNet50 Sliding Window Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed, program ended.")

if __name__ == '__main__':
    main()