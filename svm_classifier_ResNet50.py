import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50
import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('svm_classifier_resnet50.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Support Vector Machine (SVM) Hybrid Workflow
class SVMHybridTrainer:
    def __init__(self, train_dir, val_dir, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Loading data for feature extraction
        self.train_loader, self.val_loader = self._load_data(train_dir, val_dir)

        # Loading a pre-trained ResNet model as a feature extractor
        self.feature_extractor = self._load_feature_extractor()

    def _load_data(self, train_dir, val_dir):
        # This method loads and preprocesses the image data.
        # The image transformations here must be consistent with those used
        # for the validation set during the ResNet50 model training.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

        # Create data loaders. Note that shuffle=False is used because
        # the order of feature extraction needs to be consistent.
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

        return train_loader, val_loader

    def _load_feature_extractor(self):
        # We load the best model trained by the ResNet50Trainer 
        # and remove its final 'Brain' layer (The Classifier).
        logging.info("Load the best ResNet50 model as the feature extractor...")

        # Find the best model checkpoint path
        best_model_path = os.path.join('resnet50_checkpoint', 'best_resnet50_model.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found: {best_model_path}. Please run the train_resnet50_model.py first.")
        
        checkpoint = torch.load(best_model_path, map_location=self.device)

        # Rebuild the model structure and load the saved weights
        model = resnet50() # We only need the architecture, not the pre-trained weights.
        num_features = model.fc.in_features
        # The model structure here must be exactly the same as the one used during training.
        # We use nn.Sequential to ensure this.
        model.fc = nn.Sequential(
            # Even if it is not used in evaluation, it is structurally required to match the weights
            nn.Dropout(0.5),
            nn.Linear(num_features, len(checkpoint['classes']))
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # We snip off the final layer
        # We replace it with an "Identity" layer that just passes the features.
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(self.device)
        feature_extractor.eval()  # Set to evaluation mode

        logging.info("Feature extractor loaded successfully.")
        return feature_extractor

    # This function extracts features from the given data loader
    def _extract_features(self, data_loader):
        # This function takes image data, passes it through the ResNet "Eyes",
        # and collects the output features descriptions (Vector).
        features_list = []
        labels_list = []

        # We don't need to compute gradients here
        # to speed up computation and reduce memory usage.
        with torch.no_grad():
            for inputs, lbls in data_loader:
                inputs = inputs.to(self.device)
                # Get the feature vector from our modified model
                # Pass the image through our feature extractor to get the output.
                feature_output = self.feature_extractor(inputs)
                # Flatten the output and move to CPU for scikit-learn
                features_output = feature_output.view(feature_output.size(0), -1)
                features_list.append(features_output.cpu().numpy())
                labels_list.append(lbls.cpu().numpy())

        # Concatenate the features and labels frin all batches into single arrays.
        return np.concatenate(features_list), np.concatenate(labels_list)

    # Train and evaluate SVM
    def train_and_evaluate_svm(self):
        # This function orchestrates the entire workflow: extract features, train SVM, and evaluate.
        # It extracts features and then trains and tests the SVM.
        # Run Step 1: Extract features for all our data
        logging.info("Starting Step 1: Start extracting features of the training and validation sets...")
        train_features, train_labels = self._extract_features(self.train_loader)
        val_features, val_labels = self._extract_features(self.val_loader)
        logging.info(f"Extraction completed. Number of training set features: {len(train_features)}")
        logging.info(f"Extraction completed. Number of validation set features: {len(val_features)}")

        # Run Step 2: Train the SVM "Brain"
        logging.info("Starting Step 2: Training the SVM brain or classifier...")
        # We create an SVM classifier. C = 1.0 is a good default for the regularization parameter.
        # The 'kernel' can be 'linear' or 'rbf' (more flexible). We assumes that the features from the deep learning
        # model are linearly separable.
        # We use a linear kernel function, which is usually a good starting point for deep learning based features.
        svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

        # The .fit() method is the training process for the SVM
        svm_classifier.fit(train_features, train_labels)
        logging.info("SVM training completed.")

        # Run Step 3: Evaluate the new SVM model
        logging.info("Starting Step 3: Evaluating the SVM classifier on validation data...")
        # Use the trained SVM to make predictions on the validation features
        val_predictions = svm_classifier.predict(val_features)
        # Calculate the accuracy
        accuracy = accuracy_score(val_labels, val_predictions)

        print("\n" + "="*50)
        logging.info(f"ResNet50 + SVM Hybrid Model Validation Accuracy: {accuracy:.4f}")
        print("="*50 + "\n")

        return accuracy
    
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')

    try:
        classifier = SVMHybridTrainer(train_dir=train_dir, val_dir=val_dir, batch_size=32)
        classifier.train_and_evaluate_svm()
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        logging.info("Please ensure you run the train_resnet18_model.py first to create the 'best_resnet18_model.pth' file.")

if __name__ == "__main__":
    main()
