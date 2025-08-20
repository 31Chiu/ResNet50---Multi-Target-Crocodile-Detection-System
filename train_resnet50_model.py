import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime
import logging

# Configure logging to save progress and output to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_resnet50.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ResNet50Trainer:
    def __init__(self, train_dir, val_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
        # 1. Setup the main environment and hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        # 2. Define key training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 3. Set data paths
        self.train_dir = train_dir
        self.val_dir = val_dir

        # 4. Initialize image transformations
        self.train_transform, self.val_transform = self._build_transforms()

        # 5. Load the datasets
        self.train_loader, self.val_loader, self.num_classes = self._load_data()

        # 6. Build the AI model
        self.model = self._build_model()

        # 7. Define the loss function and the optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )

        # 8. Set up a learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=3
        )

        # 9. Variable to track the best performance
        self.best_acc = 0.0

    def _build_transforms(self):
        # Build data transforms for training and validation sets.
        # ImageNet statistics for normalization (Crucial for transfer learning)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Transformations for the training data to introduce variability
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Transformations for the validation data (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return train_transform, val_transform
    
    def _load_data(self):
        # Load and prepare the datasets for training and validation.
        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        logging.info(f'Number of classes: {len(train_dataset.classes)}')
        logging.info(f'Class names: {train_dataset.classes}')

        return train_loader, val_loader, len(train_dataset.classes)
    
    def _build_model(self):
        # Build the ResNet50 model using transfer learning.
        # Load a pretrained ResNet50 model
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Replace the final layer to match the number of classes in our project
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, self.num_classes)
        )

        # Send the model to the designated device (GPU or CPU)
        model = model.to(self.device)
        return model
    
    def train_epoch(self):
        # Logic for training the model for one complete pass over the training data.
        self.model.train() # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        return epoch_loss, epoch_acc.item()
    
    def validate(self):
        # Logic for evaluating the model's performance on the validation data.
        self.model.eval() # Set the model to evaluation mode
        running_loss = 0.0
        running_corrects = 0
        total = 0

        with torch.no_grad(): # Disable gradient calculation for efficiency
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            return epoch_loss, epoch_acc.item()
        
    def save_checkpoint(self, epoch, acc):
        # Save the model's state as a checkpoint file.
        checkpoint_dir = 'resnet50_checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
            'accuracy' : acc,
            'classes' : self.train_loader.dataset.classes
        }

        # Save the best performing model separately
        if acc > self.best_acc:
            self.best_acc = acc
            best_path = os.path.join(checkpoint_dir, 'best_resnet50_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f'Best model updated and saved: {best_path}')

    def train(self):
        # The main training loop that orchestrates the entire process.
        logging.info(f'Starting training on device: {self.device}')

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_acc)

            logging.info(
                f'Epoch {epoch} / {self.num_epochs} | '
                f'Train Loss: {train_loss: .4f} Acc: {train_acc: .4f} | '
                f'Val Loss: {val_loss: .4f} Acc: {val_acc: .4f}'
            )

            self.save_checkpoint(epoch, val_acc)

        logging.info(f'Training complete. Best validation accuracy: {self.best_acc: .4f}')

def main():
    # Main function to configure and run the training.
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # Define paths and parameters
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')

    # Initialize and run the trainer
    trainer = ResNet50Trainer(
        train_dir=train_dir,
        val_dir=val_dir,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    trainer.train()

if __name__ == '__main__':
    main()
