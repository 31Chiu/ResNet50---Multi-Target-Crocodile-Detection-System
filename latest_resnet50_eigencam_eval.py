import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    image_name = input("Please enter the name of the image to test (e.g., sample1.jpg): ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 严格匹配 ResNet-50 架构
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)
    
    checkpoint_path = 'resnet50_checkpoint/best_resnet50_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Weight file not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded model weights! Highest validation accuracy: {checkpoint['accuracy']:.4f}")

    target_layers = [model.layer4[-1]]

    input_dir = 'Test_Eigen-CAM_Images'
    img_path = os.path.join(input_dir, image_name) 
    if not os.path.exists(img_path):
        return

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb_img = np.array(Image.open(img_path).convert('RGB'))
    rgb_img_float = np.float32(rgb_img) / 255
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # 使用 EigenCAM
    cam = EigenCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    resized_img = cv2.resize(rgb_img_float, (224, 224))
    visualization = show_cam_on_image(resized_img, grayscale_cam, use_rgb=True)

    output_dir = 'Test_Eigen-CAM_Results'
    os.makedirs(output_dir, exist_ok=True) 
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Cropped Image")
    plt.imshow(resized_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("ResNet-50 EigenCAM Heatmap")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"resnet50_EigenCAM_{image_name}")
    plt.savefig(output_path)
    print(f"Analysis complete! Heatmap saved as {output_path}")

if __name__ == '__main__':
    main()