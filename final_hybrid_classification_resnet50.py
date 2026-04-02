import os
import cv2
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, models

# Import EigenCAM toolkit
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- ⚙️ Core Configuration Area ---
# Switched to the newly trained ResNet50 640x640 version
MODEL_TYPE = 'resnet50'  
WEIGHTS_PATH = 'resnet50_checkpoint/best_resnet50_model.pth' 

INPUT_VIDEO_PATH = 'test_video.mp4'  # Make sure this points to your BBC video
OUTPUT_VIDEO_PATH = f'Output_Classification_EigenCAM_{MODEL_TYPE}_640.mp4'
CSV_OUTPUT_PATH = f'Log_Classification_{MODEL_TYPE}_640.csv'

# Global field of view size control 🪟
CROP_SIZE = 640
RESIZE_SIZE = 680 
# -----------------------

def load_selected_model(model_type, num_classes, weights_path, device):
    """Accurately load weights 📦"""
    if model_type == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
    elif model_type == 'resnet101':
        model = models.resnet101(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError("Unsupported model type")

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Successfully loaded {model_type} weights!")
    else:
        print(f"❌ Warning: Cannot find weight file {weights_path}")
        
    return model.to(device).eval()

def get_target_layers(model, model_type):
    """Target the feature extraction layer 🎯"""
    # 🌟 Core Strategy: Uniformly lock all ResNet models to layer3 for clear spatial resolution
    if 'resnet' in model_type:
        return [model.layer3[-1]]
    else:
        print(f"⚠️ Architecture {model_type} not defined in target layers.")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting large-field-of-view heatmap inference, device: {device}")

    class_names = ['crocodile', 'non-crocodile']
    model = load_selected_model(MODEL_TYPE, len(class_names), WEIGHTS_PATH, device)

    target_layers = get_target_layers(model, MODEL_TYPE)
    cam = EigenCAM(model=model, target_layers=target_layers)

    # Expanded preprocessing pipeline 🖼️
    transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocessing purely for visual display (without normalization)
    vis_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE)
    ])

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    csv_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 1. Model inference to get confidence score
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        pred_class = class_names[predicted_idx.item()]
        conf_score = confidence.item()

        # 2. Generate EigenCAM heatmap 🔴
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # 3. Overlay the heatmap on the dynamically sized visual image
        rgb_crop = np.array(vis_transform(pil_img))
        rgb_crop_float = np.float32(rgb_crop) / 255.0
        cam_image = show_cam_on_image(rgb_crop_float, grayscale_cam, use_rgb=True)
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # 4. Dynamically calculate the center position of the original image and paste back the heatmap 🧩
        top = (height - CROP_SIZE) // 2
        left = (width - CROP_SIZE) // 2
        
        # Boundary protection: Ensure the crop area does not exceed the physical video boundaries
        if top >= 0 and left >= 0 and top+CROP_SIZE <= height and left+CROP_SIZE <= width:
            frame[top:top+CROP_SIZE, left:left+CROP_SIZE] = cam_image_bgr
            cv2.rectangle(frame, (left, top), (left+CROP_SIZE, top+CROP_SIZE), (0, 255, 255), 3)

        process_time = time.time() - start_time
        current_fps = 1.0 / process_time if process_time > 0 else 0

        csv_data.append({
            'Frame': frame_count,
            'Prediction': pred_class,
            'Confidence': round(conf_score, 4),
            'FPS': round(current_fps, 2)
        })

        text_color = (0, 0, 255) if pred_class == 'crocodile' else (0, 255, 0)
        cv2.putText(frame, f"Model: {MODEL_TYPE} 640x640 (EigenCAM)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Result: {pred_class} ({conf_score:.2f})", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out_video.write(frame)
        
        if frame_count % 50 == 0:
            print(f"⏳ Progress: {frame_count}/{total_frames} frames... (FPS: {current_fps:.1f})")

    cap.release()
    out_video.release()
    
    df = pd.DataFrame(csv_data)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\n🎉 Processing complete! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()