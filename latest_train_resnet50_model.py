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
from sklearn.metrics import average_precision_score
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('latest_training_resnet50.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ResNet50Trainer:
    def __init__(self, train_dir, val_dir, num_epochs=50, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # 🌟 核心修改 1：设置物理批次与梯度累加步数
        self.physical_batch_size = 8   # 显卡实际每次处理 8 张图 (防 OOM)
        self.accumulation_steps = 4    # 累加 4 次 (8 * 4 = 32 逻辑批次)

        self.train_dir = train_dir
        self.val_dir = val_dir

        self.train_transform, self.val_transform = self._build_transforms()

        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.physical_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )

        val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transform)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.physical_batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )

        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3, verbose=True)

        self.best_acc = 0.0

    def _build_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # 🌟 同步 640x640 视野
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(640), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30), 
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random') 
        ])

        val_transform = transforms.Compose([
            transforms.Resize(680),            
            transforms.CenterCrop(640),        
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return train_transform, val_transform

    def _build_model(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )
        return model.to(self.device)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # 🌟 确保在 epoch 开始前梯度已清零
        self.optimizer.zero_grad()

        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 🌟 核心修改 3：按累加步数缩小 loss
            loss = loss / self.accumulation_steps
            loss.backward()

            # 🌟 核心修改 4：达到步数，或到达最后一个 batch 时，更新权重
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 将 loss 还原回真实比例用于统计
            running_loss += (loss.item() * self.accumulation_steps) * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            # 对于二分类，提取正类的概率用于 mAP
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_preds.extend(probs[:, 1].detach().cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # 计算精确率、召回率和 mAP 
        # (这里简化处理，假设类别 1 是目标类。在实际中需确保这与你的特定类索引匹配)
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score
        epoch_prec = precision_score(all_labels, preds_binary, average='macro', zero_division=0)
        epoch_rec = recall_score(all_labels, preds_binary, average='macro', zero_division=0)
        epoch_map = average_precision_score(all_labels, all_preds)

        return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_map

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_preds.extend(probs[:, 1].cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score
        epoch_prec = precision_score(all_labels, preds_binary, average='macro', zero_division=0)
        epoch_rec = recall_score(all_labels, preds_binary, average='macro', zero_division=0)
        epoch_map = average_precision_score(all_labels, all_preds)

        return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_map

    def save_checkpoint(self, epoch, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            checkpoint_dir = 'resnet50_checkpoint'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_resnet50_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': val_acc
            }, checkpoint_path)
            logging.info(f'Best model updated and saved: {checkpoint_path}')

    def run(self):
        logging.info(f'Starting training on device: {self.device}')

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc, train_prec, train_rec, train_map = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_map = self.validate()
            
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            logging.info(
                f'Epoch {epoch:02d}/{self.num_epochs} [LR: {current_lr:.6f}] | '
                f'Train -> Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} Rec: {train_rec:.4f} mAP: {train_map:.4f} | '
                f'Val -> Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} Rec: {val_rec:.4f} mAP: {val_map:.4f}'
            )
            self.save_checkpoint(epoch, val_acc)

        logging.info(f'Training complete. Best validation accuracy: {self.best_acc: .4f}')

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')

    trainer = ResNet50Trainer(train_dir=train_dir, val_dir=val_dir, num_epochs=50)
    trainer.run()

if __name__ == '__main__':
    main()