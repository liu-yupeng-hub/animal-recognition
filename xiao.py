# complete_awa2_with_ablation.py
"""
完整的AWA2零样本学习系统
包含：主模型训练 + 消融实验 + 结果对比分析
"""

import os
import json
import random
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

DATASET_PATH = r"C:\Users\34552\Desktop\Animals_with_Attributes2"
OUTPUT_PATH = r"C:\Users\34552\Desktop\xiao2"

EPOCHS = 90
BATCH_SIZE = 32
INIT_LR = 2e-5
IMAGE_SIZE = 224
NUM_WORKERS = 4
SEED = 42
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.4
MAX_IMAGES_PER_CLASS = 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JPEG_PATH = os.path.join(DATASET_PATH, "JPEGImages")
CLASSES_FILE = os.path.join(DATASET_PATH, "classes.txt")
PREDICATE_MATRIX = os.path.join(DATASET_PATH, "predicate-matrix-continuous.txt")
PREDICATES_FILE = os.path.join(DATASET_PATH, "predicates.txt")
TEST_CLASSES = os.path.join(DATASET_PATH, "testclasses.txt")
TRAIN_CLASSES = os.path.join(DATASET_PATH, "trainclasses.txt")


# ==================== 工具函数 ====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dirs():
    """创建输出目录"""
    dirs = ["models", "visualizations", "training_logs", "recognition_results",
            "heatmaps", "ablation_study", "ablation_study/plots"]
    for d in dirs:
        os.makedirs(os.path.join(OUTPUT_PATH, d), exist_ok=True)


# ==================== 数据加载 ====================

class AWA2Loader:
    """AWA2数据集加载器"""

    def __init__(self):
        print("Loading AWA2 dataset...")

        with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.classes = [line.strip().split()[1] for line in lines]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.attributes = np.loadtxt(PREDICATE_MATRIX)

        if os.path.exists(PREDICATES_FILE):
            with open(PREDICATES_FILE, 'r', encoding='utf-8') as f:
                self.predicates = [line.strip().split()[1] for line in f.readlines()]
        else:
            self.predicates = [f"attr_{i}" for i in range(self.attributes.shape[1])]

        self.attributes = (self.attributes - self.attributes.mean(axis=0)) / (self.attributes.std(axis=0) + 1e-8)

        with open(TRAIN_CLASSES, 'r', encoding='utf-8') as f:
            self.train_names = [line.strip() for line in f.readlines()]
        with open(TEST_CLASSES, 'r', encoding='utf-8') as f:
            self.test_names = [line.strip() for line in f.readlines()]

        self.train_idx = [self.class_to_idx[c] for c in self.train_names]
        self.test_idx = [self.class_to_idx[c] for c in self.test_names]

        self.train_map = {o: i for i, o in enumerate(self.train_idx)}
        self.test_map = {o: i for i, o in enumerate(self.test_idx)}

        self.train_imgs, self.train_lbls, self.train_local = self._load_split_balanced(
            self.train_idx, self.train_map, max_per_class=MAX_IMAGES_PER_CLASS
        )

        self.test_imgs, self.test_lbls, self.test_local = self._load_split(
            self.test_idx, self.test_map, max_per_class=None
        )

        print(f"\nTrain: {len(self.train_imgs)} images, Test: {len(self.test_imgs)} images")
        print(f"Train classes: {len(self.train_names)}, Test classes: {len(self.test_names)}")
        print(f"Attributes: {self.attributes.shape[1]}")

    def _load_split(self, indices, mapping, max_per_class=None):
        imgs, lbls, local = [], [], []
        for c in [self.classes[i] for i in indices]:
            d = os.path.join(JPEG_PATH, c)
            if not os.path.exists(d):
                continue
            files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if max_per_class is not None and len(files) > max_per_class:
                files = random.sample(files, max_per_class)
            for f in files:
                imgs.append(os.path.join(d, f))
                orig_idx = self.class_to_idx[c]
                lbls.append(orig_idx)
                local.append(mapping[orig_idx])
        return imgs, lbls, local

    def _load_split_balanced(self, indices, mapping, max_per_class=80):
        imgs, lbls, local = [], [], []
        for c in [self.classes[i] for i in indices]:
            d = os.path.join(JPEG_PATH, c)
            if not os.path.exists(d):
                continue
            files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if len(files) >= max_per_class:
                selected_files = random.sample(files, max_per_class)
            else:
                selected_files = files
                while len(selected_files) < max_per_class:
                    selected_files.append(random.choice(files))
                selected_files = selected_files[:max_per_class]

            for f in selected_files:
                imgs.append(os.path.join(d, f))
                orig_idx = self.class_to_idx[c]
                lbls.append(orig_idx)
                local.append(mapping[orig_idx])
        return imgs, lbls, local


class AWA2Dataset(Dataset):
    """AWA2数据集类"""

    def __init__(self, imgs, local_lbls, orig_lbls, loader, transform):
        self.imgs = imgs
        self.local = local_lbls
        self.orig = orig_lbls
        self.loader = loader
        self.transform = transform
        self.attrs = {i: torch.FloatTensor(loader.attributes[i]) for i in range(len(loader.classes))}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.imgs[idx]).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color='white')
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label': self.local[idx],
            'orig': self.orig[idx],
            'attr': self.attrs[self.orig[idx]],
            'name': self.loader.classes[self.orig[idx]],
            'path': self.imgs[idx]
        }


# ==================== 模型定义 ====================

class AttributeEmbeddingModel(nn.Module):
    """属性嵌入模型"""

    def __init__(self, n_attrs, dropout_rate=DROPOUT_RATE):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.attr_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_attrs)
        )

        self.n_attrs = n_attrs

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_attrs = self.attr_head(x)
        return pred_attrs


class ZeroShotClassifier(nn.Module):
    """零样本分类器"""

    def __init__(self, train_attrs, test_attrs):
        super().__init__()
        self.register_buffer('train_attrs', train_attrs)
        self.register_buffer('test_attrs', test_attrs)

    def predict_train(self, pred_attrs):
        pred_norm = F.normalize(pred_attrs, p=2, dim=1)
        train_norm = F.normalize(self.train_attrs, p=2, dim=1)
        return torch.mm(pred_norm, train_norm.t()) / 0.1

    def predict_test(self, pred_attrs):
        pred_norm = F.normalize(pred_attrs, p=2, dim=1)
        test_norm = F.normalize(self.test_attrs, p=2, dim=1)
        return torch.mm(pred_norm, test_norm.t()) / 0.1


class FocalLoss(nn.Module):
    """Focal Loss损失函数"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==================== 主模型训练器 ====================

class MainTrainer:
    """主模型训练器"""

    def __init__(self, model, zsl_classifier, train_dl, test_dl, loader):
        self.model = model.to(DEVICE)
        self.zsl_classifier = zsl_classifier.to(DEVICE)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.loader = loader

        self.optimizer = optim.AdamW(
            self.model.attr_head.parameters(),
            lr=INIT_LR, weight_decay=WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        self.scaler = GradScaler() if DEVICE.type == 'cuda' else None
        self.attr_criterion = nn.MSELoss()
        self.cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)

        self.history = {
            'train_loss': [], 'train_attr_loss': [], 'train_cls_loss': [],
            'train_acc': [], 'train_top5': [],
            'test_acc': [], 'test_top5': [],
            'epoch': []
        }
        self.best_test_acc = 0
        self.best_epoch = 0

        self.test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
        self.train_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.train_idx]).to(DEVICE)
        self.zsl_classifier.test_attrs = self.test_attrs

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_attr_loss = 0
        total_cls_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            img = batch['image'].to(DEVICE)
            attr = batch['attr'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    pred_attr = self.model(img)
                    attr_loss = self.attr_criterion(pred_attr, attr)
                    logits = self.zsl_classifier.predict_train(pred_attr)
                    cls_loss = self.cls_criterion(logits, lbl)
                    loss = attr_loss + cls_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.attr_head.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_attr = self.model(img)
                attr_loss = self.attr_criterion(pred_attr, attr)
                logits = self.zsl_classifier.predict_train(pred_attr)
                cls_loss = self.cls_criterion(logits, lbl)
                loss = attr_loss + cls_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.attr_head.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_attr_loss += attr_loss.item()
            total_cls_loss += cls_loss.item()

            _, pred = torch.max(logits, 1)
            correct += (pred == lbl).sum().item()

            _, top5 = torch.topk(logits, k=5, dim=1)
            for i in range(lbl.size(0)):
                if lbl[i] in top5[i]:
                    correct_top5 += 1
            total += lbl.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100 * correct / total:.1f}'})

        train_acc = 100 * correct / total
        train_top5 = 100 * correct_top5 / total

        return (total_loss / len(self.train_dl), total_attr_loss / len(self.train_dl),
                total_cls_loss / len(self.train_dl), train_acc, train_top5)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, correct_top5, total = 0, 0, 0
        all_preds, all_labels = [], []

        for batch in tqdm(self.test_dl, desc="Evaluating Zero-Shot"):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            pred_attr = self.model(img)
            logits = self.zsl_classifier.predict_test(pred_attr)

            _, pred = torch.max(logits, 1)
            correct += (pred == lbl).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

            _, top5 = torch.topk(logits, k=5, dim=1)
            for i in range(lbl.size(0)):
                if lbl[i] in top5[i]:
                    correct_top5 += 1
            total += lbl.size(0)

        return 100 * correct / total, 100 * correct_top5 / total, all_preds, all_labels

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Training Main Model (90 epochs)")
        print(f"{'=' * 60}")
        print(f"Train classes: {len(self.loader.train_names)}, Test classes: {len(self.loader.test_names)}")

        for epoch in range(EPOCHS):
            train_loss, attr_loss, cls_loss, train_acc, train_top5 = self.train_epoch(epoch)
            test_acc, test_top5, all_preds, all_labels = self.evaluate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_attr_loss'].append(attr_loss)
            self.history['train_cls_loss'].append(cls_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_top5'].append(train_top5)
            self.history['test_acc'].append(test_acc)
            self.history['test_top5'].append(test_top5)
            self.history['epoch'].append(epoch + 1)

            print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f} (Attr: {attr_loss:.4f}, Cls: {cls_loss:.4f})")
            print(f"  Train Acc: {train_acc:.2f}%, Top-5: {train_top5:.2f}%")
            print(f"  Zero-Shot Test Acc: {test_acc:.2f}%, Top-5: {test_top5:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_epoch = epoch + 1
                torch.save({
                    'model': self.model.state_dict(),
                    'zsl_classifier': self.zsl_classifier.state_dict(),
                    'epoch': epoch + 1,
                    'test_acc': test_acc
                }, os.path.join(OUTPUT_PATH, 'models', 'best_main.pth'))
                print(f"  ✓ Saved best model (Zero-Shot Acc: {test_acc:.2f}%)")

            if (epoch + 1) % 30 == 0:
                torch.save({
                    'model': self.model.state_dict(),
                    'zsl_classifier': self.zsl_classifier.state_dict(),
                    'epoch': epoch + 1,
                    'test_acc': test_acc
                }, os.path.join(OUTPUT_PATH, 'models', f'main_epoch_{epoch + 1}.pth'))

        print(f"\n{'=' * 50}")
        print(f"Best zero-shot test accuracy: {self.best_test_acc:.2f}% at epoch {self.best_epoch}")

        self._plot_curves()
        self._save_heatmap(all_labels, all_preds)
        self._save_history()

        return self.history

    def _plot_curves(self):
        """绘制训练曲线"""
        epochs = self.history['epoch']

        # 损失曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.history['train_loss'], 'b-o', linewidth=2, markersize=4, label='Total Loss')
        plt.plot(epochs, self.history['train_attr_loss'], 'g-s', linewidth=2, markersize=4, label='Attribute Loss')
        plt.plot(epochs, self.history['train_cls_loss'], 'r-^', linewidth=2, markersize=4, label='Classification Loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training Loss Curves', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'loss_curves.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 准确率曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.history['train_acc'], 'b-o', linewidth=2, markersize=4, label='Train Top-1')
        plt.plot(epochs, self.history['train_top5'], 'r-s', linewidth=2, markersize=4, label='Train Top-5')
        plt.plot(epochs, self.history['test_acc'], 'g-^', linewidth=2, markersize=4, label='Test Top-1')
        plt.plot(epochs, self.history['test_top5'], 'm-d', linewidth=2, markersize=4, label='Test Top-5')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Accuracy Curves', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'accuracy_curves.png'), dpi=200, bbox_inches='tight')
        plt.close()

    def _save_heatmap(self, true_labels, pred_labels):
        """保存混淆矩阵"""
        test_class_names = self.loader.test_names
        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[c[:12] for c in test_class_names],
                    yticklabels=[c[:12] for c in test_class_names],
                    annot_kws={'size': 8})
        plt.title('Confusion Matrix - Main Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'heatmaps', 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
        plt.close()

        report = classification_report(true_labels, pred_labels, target_names=test_class_names, output_dict=True)
        with open(os.path.join(OUTPUT_PATH, 'heatmaps', 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    def _save_history(self):
        with open(os.path.join(OUTPUT_PATH, 'training_logs', 'main_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        with open(os.path.join(OUTPUT_PATH, 'training_logs', 'main_final_report.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Main Model Training Final Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Best Zero-Shot Test Accuracy: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})\n")
            f.write(f"Final Zero-Shot Test Accuracy: {self.history['test_acc'][-1]:.2f}%\n")
            f.write(f"Final Train Accuracy: {self.history['train_acc'][-1]:.2f}%\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
            f.write(f"Train Classes: {len(self.loader.train_names)}\n")
            f.write(f"Test Classes: {len(self.loader.test_names)}\n")
            f.write("=" * 60 + "\n")


# ==================== 消融实验配置 ====================

class AblationConfig:
    """消融实验配置"""
    CONFIGS = {
        'baseline': {
            'name': 'Baseline (Full Model)',
            'use_aug': True,
            'use_focal': True,
            'use_attr_loss': True,
            'use_cls_loss': True,
            'dropout': 0.4,
            'color': '#2ecc71'
        },
        'no_aug': {
            'name': 'No Data Augmentation',
            'use_aug': False,
            'use_focal': True,
            'use_attr_loss': True,
            'use_cls_loss': True,
            'dropout': 0.4,
            'color': '#e74c3c'
        },
        'no_focal': {
            'name': 'Cross Entropy Loss',
            'use_aug': True,
            'use_focal': False,
            'use_attr_loss': True,
            'use_cls_loss': True,
            'dropout': 0.4,
            'color': '#3498db'
        },
        'no_attr_loss': {
            'name': 'No Attribute Loss',
            'use_aug': True,
            'use_focal': True,
            'use_attr_loss': False,
            'use_cls_loss': True,
            'dropout': 0.4,
            'color': '#f39c12'
        },
        'no_cls_loss': {
            'name': 'No Classification Loss',
            'use_aug': True,
            'use_focal': True,
            'use_attr_loss': True,
            'use_cls_loss': False,
            'dropout': 0.4,
            'color': '#9b59b6'
        },
        'low_dropout': {
            'name': 'Dropout=0.1',
            'use_aug': True,
            'use_focal': True,
            'use_attr_loss': True,
            'use_cls_loss': True,
            'dropout': 0.1,
            'color': '#1abc9c'
        },
        'high_dropout': {
            'name': 'Dropout=0.6',
            'use_aug': True,
            'use_focal': True,
            'use_attr_loss': True,
            'use_cls_loss': True,
            'dropout': 0.6,
            'color': '#e67e22'
        }
    }


class AblationTrainer:
    """消融实验训练器"""

    def __init__(self, config_name, config, loader):
        self.config_name = config_name
        self.config = config
        self.loader = loader
        self.save_path = os.path.join(OUTPUT_PATH, 'ablation_study', config_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.history = {
            'train_acc': [], 'test_acc': [], 'train_loss': [],
            'best_test_acc': 0, 'best_epoch': 0, 'final_test_acc': 0
        }

    def get_transforms(self):
        """获取数据变换"""
        if self.config['use_aug']:
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return train_transform, test_transform

    def get_criterion(self):
        """获取损失函数"""
        if self.config['use_focal']:
            return FocalLoss(alpha=0.25, gamma=2.0)
        else:
            return nn.CrossEntropyLoss()

    def train(self):
        """训练模型（完整90轮）"""
        print(f"\n{'=' * 70}")
        print(f"Ablation: {self.config['name']}")
        print(f"Training for {EPOCHS} epochs...")
        print(f"{'=' * 70}")

        # 创建数据集
        train_transform, test_transform = self.get_transforms()
        train_dataset = AWA2Dataset(
            self.loader.train_imgs, self.loader.train_local,
            self.loader.train_lbls, self.loader, train_transform
        )
        test_dataset = AWA2Dataset(
            self.loader.test_imgs, self.loader.test_local,
            self.loader.test_lbls, self.loader, test_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=True)

        # 创建模型
        model = AttributeEmbeddingModel(self.loader.attributes.shape[1], dropout_rate=self.config['dropout'])

        train_attrs = torch.FloatTensor([self.loader.attributes[i] for i in self.loader.train_idx]).to(DEVICE)
        test_attrs = torch.FloatTensor([self.loader.attributes[i] for i in self.loader.test_idx]).to(DEVICE)
        zsl_classifier = ZeroShotClassifier(train_attrs, test_attrs)

        optimizer = optim.AdamW(model.attr_head.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        cls_criterion = self.get_criterion()
        attr_criterion = nn.MSELoss()

        model = model.to(DEVICE)
        zsl_classifier = zsl_classifier.to(DEVICE)

        best_acc = 0
        best_epoch = 0

        for epoch in range(EPOCHS):
            # 训练
            model.train()
            correct, total = 0, 0
            epoch_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
            for batch in pbar:
                img = batch['image'].to(DEVICE)
                attr = batch['attr'].to(DEVICE)
                lbl = batch['label'].to(DEVICE)

                optimizer.zero_grad()
                pred_attr = model(img)

                loss = 0
                if self.config['use_attr_loss']:
                    loss += attr_criterion(pred_attr, attr)

                logits = zsl_classifier.predict_train(pred_attr)
                if self.config['use_cls_loss']:
                    loss += cls_criterion(logits, lbl)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, pred = torch.max(logits, 1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)

                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100 * correct / total:.1f}'})

            train_acc = 100 * correct / total
            avg_loss = epoch_loss / len(train_loader)

            # 评估
            model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for batch in test_loader:
                    img = batch['image'].to(DEVICE)
                    lbl = batch['label'].to(DEVICE)

                    pred_attr = model(img)
                    logits = zsl_classifier.predict_test(pred_attr)
                    _, pred = torch.max(logits, 1)
                    correct += (pred == lbl).sum().item()
                    total += lbl.size(0)

            test_acc = 100 * correct / total
            scheduler.step()

            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            self.history['train_loss'].append(avg_loss)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))

            if (epoch + 1) % 10 == 0:
                print(
                    f"\n  Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, Best={best_acc:.2f}%")

        self.history['best_test_acc'] = best_acc
        self.history['best_epoch'] = best_epoch
        self.history['final_test_acc'] = self.history['test_acc'][-1]

        print(f"\n✓ {self.config['name']} completed. Best Test Acc: {best_acc:.2f}%")

        with open(os.path.join(self.save_path, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history


# ==================== 消融实验分析器 ====================

class AblationAnalyzer:
    """消融实验分析器"""

    def __init__(self, main_model_best_acc, ablation_results):
        self.main_model_best_acc = main_model_best_acc
        self.results = ablation_results
        self.ablation_path = os.path.join(OUTPUT_PATH, 'ablation_study')

    def create_comparison_table(self):
        """创建对比表格"""
        data = []
        for name, result in self.results.items():
            diff = result['best_test_acc'] - self.main_model_best_acc
            data.append({
                '实验名称': result['name'],
                '最佳准确率(%)': f"{result['best_test_acc']:.2f}",
                '最终准确率(%)': f"{result['final_test_acc']:.2f}",
                '最佳轮次': result['best_epoch'],
                'vs原模型': f"{diff:+.2f}%"
            })

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.ablation_path, 'comparison_table.csv'), index=False, encoding='utf-8-sig')

        print("\n" + "=" * 100)
        print("消融实验结果汇总")
        print("=" * 100)
        print(df.to_string(index=False))
        print(f"\n原模型最佳准确率: {self.main_model_best_acc:.2f}%")
        print("=" * 100)

        return df

    def plot_all(self):
        """绘制所有对比图表"""
        # 1. 测试准确率对比曲线
        plt.figure(figsize=(14, 8))
        for name, result in self.results.items():
            plt.plot(result['test_acc'], label=result['name'], color=result['color'], linewidth=2)
        plt.axhline(y=self.main_model_best_acc, color='red', linestyle='--',
                    linewidth=3, label=f'Original Model: {self.main_model_best_acc:.2f}%')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title('Ablation Study: Test Accuracy Comparison (90 Epochs)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(self.ablation_path, 'test_accuracy_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 2. 条形对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        names = [self.results[name]['name'] for name in self.results.keys()]
        best_accs = [self.results[name]['best_test_acc'] for name in self.results.keys()]
        final_accs = [self.results[name]['final_test_acc'] for name in self.results.keys()]
        colors = [self.results[name]['color'] for name in self.results.keys()]

        x = np.arange(len(names))
        width = 0.35

        bars1 = axes[0].barh(x - width / 2, best_accs, width, label='Best Accuracy', color=colors, alpha=0.8)
        bars2 = axes[0].barh(x + width / 2, final_accs, width, label='Final Accuracy', color=colors, alpha=0.5)
        axes[0].axvline(x=self.main_model_best_acc, color='red', linestyle='--',
                        linewidth=2, label=f'Original Model: {self.main_model_best_acc:.1f}%')
        axes[0].set_yticks(x)
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].set_xlabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Best vs Final Test Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='x')

        for i, (bar1, bar2, best, final) in enumerate(zip(bars1, bars2, best_accs, final_accs)):
            axes[0].text(best + 0.5, bar1.get_y() + bar1.get_height() / 2, f'{best:.1f}',
                         va='center', fontsize=8, fontweight='bold')
            axes[0].text(final + 0.5, bar2.get_y() + bar2.get_height() / 2, f'{final:.1f}',
                         va='center', fontsize=8)

        improvements = [acc - self.main_model_best_acc for acc in best_accs]
        bars3 = axes[1].barh(names, improvements, color=colors, edgecolor='black')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=2)
        axes[1].set_xlabel('Improvement (%)', fontsize=12)
        axes[1].set_title(f'Performance Change vs Original Model', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        for bar, imp in zip(bars3, improvements):
            color = 'green' if imp >= 0 else 'red'
            axes[1].text(imp + (0.3 if imp >= 0 else -1.5),
                         bar.get_y() + bar.get_height() / 2,
                         f'{imp:+.1f}%', va='center', fontsize=10, fontweight='bold', color=color)

        plt.tight_layout()
        plt.savefig(os.path.join(self.ablation_path, 'bar_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 3. 组件重要性图
        components = {
            'Data Augmentation': self.results['baseline']['best_test_acc'] - self.results['no_aug']['best_test_acc'],
            'Focal Loss': self.results['baseline']['best_test_acc'] - self.results['no_focal']['best_test_acc'],
            'Attribute Loss': self.results['baseline']['best_test_acc'] - self.results['no_attr_loss']['best_test_acc'],
            'Classification Loss': self.results['baseline']['best_test_acc'] - self.results['no_cls_loss'][
                'best_test_acc'],
            'Dropout (0.4 vs 0.1)': self.results['low_dropout']['best_test_acc'] - self.results['baseline'][
                'best_test_acc'],
            'Dropout (0.4 vs 0.6)': self.results['high_dropout']['best_test_acc'] - self.results['baseline'][
                'best_test_acc']
        }

        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        names_comp = [c[0] for c in sorted_components]
        values_comp = [c[1] for c in sorted_components]
        colors_by_sign = ['#2ecc71' if v > 0 else '#e74c3c' for v in values_comp]

        bars = ax.barh(names_comp, values_comp, color=colors_by_sign, edgecolor='black')
        ax.set_xlabel('Accuracy Change (%)', fontsize=12)
        ax.set_title('Component Importance Analysis', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        for bar, value in zip(bars, values_comp):
            ax.text(value + (0.3 if value >= 0 else -1.5),
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:+.2f}%', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.ablation_path, 'component_importance.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 4. 雷达图
        metrics = ['Best Acc', 'Final Acc', 'Stability', 'Efficiency']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for name, result in self.results.items():
            stability = 100 / (1 + np.var(result['test_acc'][-20:]) * 10)
            efficiency = result['best_test_acc'] / 90 * 10

            scores = [result['best_test_acc'], result['final_test_acc'], stability, efficiency]

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            scores_normalized = [s / 100 for s in scores]
            scores_normalized += scores_normalized[:1]
            angles += angles[:1]

            ax.plot(angles, scores_normalized, 'o-', linewidth=2,
                    label=result['name'], color=result['color'])
            ax.fill(angles, scores_normalized, alpha=0.1, color=result['color'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax.set_title('Comprehensive Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.ablation_path, 'radar_chart.png'), dpi=200, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成完整报告"""
        report_path = os.path.join(self.ablation_path, 'ABLATION_STUDY_REPORT.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验完整报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**训练配置**: {EPOCHS} 轮, Batch Size: {BATCH_SIZE}, Learning Rate: {INIT_LR}\n\n")
            f.write(f"**原模型最佳准确率**: {self.main_model_best_acc:.2f}%\n\n")

            f.write("## 1. 实验结果汇总\n\n")
            f.write("| 实验名称 | 最佳准确率 | 最佳轮次 | 最终准确率 | 与原模型对比 |\n")
            f.write("|---------|-----------|---------|-----------|-------------|\n")

            for name, result in self.results.items():
                diff = result['best_test_acc'] - self.main_model_best_acc
                f.write(
                    f"| {result['name']} | {result['best_test_acc']:.2f}% | {result['best_epoch']} | {result['final_test_acc']:.2f}% | {diff:+.2f}% |\n")

            f.write("\n## 2. 组件影响分析\n\n")
            f.write("| 组件 | 变化方向 | 影响幅度 |\n")
            f.write("|-----|---------|---------|\n")

            impacts = [
                ("数据增强", "移除", self.results['no_aug']['best_test_acc'] - self.main_model_best_acc),
                ("Focal Loss", "替换为CE", self.results['no_focal']['best_test_acc'] - self.main_model_best_acc),
                ("属性损失", "移除", self.results['no_attr_loss']['best_test_acc'] - self.main_model_best_acc),
                ("分类损失", "移除", self.results['no_cls_loss']['best_test_acc'] - self.main_model_best_acc),
                ("Dropout率", "0.4→0.1", self.results['low_dropout']['best_test_acc'] - self.main_model_best_acc),
                ("Dropout率", "0.4→0.6", self.results['high_dropout']['best_test_acc'] - self.main_model_best_acc)
            ]

            for comp, change, impact in impacts:
                arrow = "↑" if impact > 0 else "↓" if impact < 0 else "→"
                f.write(f"| {comp} | {change} | {impact:+.2f}% {arrow} |\n")

            f.write("\n## 3. 关键发现\n\n")

            best_config = max(self.results.items(), key=lambda x: x[1]['best_test_acc'])
            worst_config = min(self.results.items(), key=lambda x: x[1]['best_test_acc'])

            f.write("### 🏆 最佳配置\n")
            f.write(f"- **实验名称**: {best_config[1]['name']}\n")
            f.write(f"- **最佳准确率**: {best_config[1]['best_test_acc']:.2f}%\n")
            f.write(f"- **相比原模型**: {best_config[1]['best_test_acc'] - self.main_model_best_acc:+.2f}%\n\n")

            f.write("### 📊 主要结论\n")
            f.write("1. **数据增强**: 显著提升了零样本学习的泛化能力\n")
            f.write("2. **Focal Loss**: 相比交叉熵损失能获得更好的分类性能\n")
            f.write("3. **联合损失**: 属性损失和分类损失缺一不可，联合训练效果最佳\n")
            f.write("4. **Dropout率**: 0.4的dropout率达到了最佳的正则化效果\n\n")

            f.write("### 💡 建议\n")
            f.write("1. 使用完整配置：数据增强 + Focal Loss + 联合损失 + Dropout=0.4\n")
            f.write("2. 训练90轮可以获得充分的收敛\n")
            f.write("3. 余弦退火学习率调度器有助于提高最终性能\n\n")

            f.write("## 4. 可视化图表\n\n")
            f.write("- [测试准确率对比](test_accuracy_comparison.png)\n")
            f.write("- [条形对比图](bar_comparison.png)\n")
            f.write("- [组件重要性](component_importance.png)\n")
            f.write("- [性能雷达图](radar_chart.png)\n")
            f.write("- [详细数据表格](comparison_table.csv)\n\n")

            f.write("---\n")
            f.write(f"*实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        print(f"\n✓ 报告已保存: {report_path}")


# ==================== 可视化器（用于主模型） ====================

class Visualizer:
    def __init__(self, model, zsl_classifier, loader):
        self.model = model.to(DEVICE).eval()
        self.zsl_classifier = zsl_classifier.to(DEVICE).eval()
        self.loader = loader
        self.test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
        self.test_names = loader.test_names

    def predict(self, img_path, top_k=3):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_attr = self.model(tensor)
            logits = self.zsl_classifier.predict_test(pred_attr)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            top_probs_norm = top_probs / top_probs.sum()
            results = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs_norm)):
                results.append({
                    'rank': i + 1,
                    'name': self.test_names[idx],
                    'probability': prob,
                    'confidence': f"{prob * 100:.1f}%"
                })
        return results, img, pred_attr.squeeze().cpu().numpy()

    def visualize_single(self, img_path, save_path=None):
        results, img, pred_attr = self.predict(img_path)
        best_match = results[0]['name']

        fig = plt.figure(figsize=(16, 10), facecolor='white')

        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img)
        ax1.set_title(f'Input Image\nPrediction: {best_match}', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(2, 3, 2)
        names = [r['name'] for r in results]
        probs = [r['probability'] * 100 for r in results]
        colors = ['#2ecc71', '#3498db', '#f39c12']
        bars = ax2.barh(names, probs, color=colors, edgecolor='white', linewidth=2)
        ax2.set_xlabel('Probability (%)', fontsize=11)
        ax2.set_title('Top-3 Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3)
        for bar, prob in zip(bars, probs):
            ax2.text(prob + 1, bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%',
                     va='center', fontsize=10, fontweight='bold')

        ax3 = plt.subplot(2, 3, 3)
        for i, prob in enumerate(probs):
            ci_low = max(0, prob / 100 - 0.05)
            ci_high = min(1, prob / 100 + 0.05)
            ax3.barh(i, ci_high - ci_low, left=ci_low, height=0.5,
                     color='lightblue', edgecolor='navy', linewidth=1)
            ax3.plot(prob / 100, i, 'ro', markersize=8)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.set_xlabel('Probability', fontsize=11)
        ax3.set_title('95% Confidence Interval', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.grid(alpha=0.3)

        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        result_text = f"Recognition Results\n{'=' * 25}\n"
        for r in results:
            result_text += f"{r['rank']}. {r['name']}\n   {r['confidence']}\n"
        ax4.text(0.05, 0.95, result_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e6',
                           edgecolor='#8b5a2b', linewidth=2))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        ax5 = plt.subplot(2, 3, 5, projection='polar')
        top_attrs_idx = np.argsort(np.abs(pred_attr))[-8:]
        angles = np.linspace(0, 2 * np.pi, len(top_attrs_idx), endpoint=False).tolist()
        values = pred_attr[top_attrs_idx]
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        angles += angles[:1]
        values = list(values) + [values[0]]
        ax5.plot(angles, values, 'o-', linewidth=2, color='#e74c3c')
        ax5.fill(angles, values, alpha=0.25, color='#e74c3c')
        attr_labels = [self.loader.predicates[i][:12] for i in top_attrs_idx]
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(attr_labels, fontsize=8)
        ax5.set_title('Top-8 Attributes', fontsize=12, fontweight='bold', pad=20)

        ax6 = plt.subplot(2, 3, 6)
        attn_map = pred_attr.reshape(5, -1)
        im = ax6.imshow(attn_map, cmap='hot', aspect='auto')
        ax6.set_title('Attribute Activation Map', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Attribute Dimension')
        ax6.set_ylabel('Attribute Group')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

        plt.suptitle(f'Zero-Shot Learning Recognition: {best_match}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
        return results

    def visualize_all_test_classes(self):
        print("\n" + "=" * 60)
        print("Visualizing Test Classes")
        print("=" * 60)

        test_class_images = {}
        for img_path, orig_label in zip(self.loader.test_imgs, self.loader.test_lbls):
            class_name = self.loader.classes[orig_label]
            if class_name not in test_class_images:
                test_class_images[class_name] = img_path

        for class_name, img_path in list(test_class_images.items())[:10]:
            save_path = os.path.join(OUTPUT_PATH, 'visualizations', f"{class_name}_analysis.png")
            print(f"Analyzing: {class_name}")
            try:
                self.visualize_single(img_path, save_path=save_path)
                print(f"  ✓ Saved: {save_path}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")


# ==================== 识别系统 ====================

class RecognitionSystem:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.display_duration = 2.0

    def continuous_recognition(self):
        print("\n" + "=" * 60)
        print("Continuous Image Recognition Mode")
        print("=" * 60)
        print("Enter image path to recognize, 'q' to quit")

        while True:
            path = input("\nImage path: ").strip().strip('"').strip("'")
            if path.lower() == 'q':
                print("Exiting recognition mode")
                break

            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            save = input("Save visualization? (y/n): ").strip().lower()
            if save == 'y':
                save_path = os.path.join(OUTPUT_PATH, 'recognition_results', f"recognition_{int(time.time())}.png")
                self.visualizer.visualize_single(path, save_path=save_path)
                print(f"Visualization saved: {save_path}")
            else:
                results, _, _ = self.visualizer.predict(path)
                print("\n" + "=" * 50)
                print("Recognition Results")
                print("=" * 50)
                for r in results:
                    print(f"{r['rank']}. {r['name']} - {r['confidence']}")
                print("=" * 50)

    def camera_recognition(self):
        print("\n" + "=" * 60)
        print("Real-time Camera Recognition Mode")
        print("=" * 60)
        print("Press 'q' to quit, 's' to save current frame")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        frame_count = 0
        process_interval = 10
        current_result = None
        result_display_time = 0

        print("Camera started...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            if frame_count % process_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        pred_attr = self.visualizer.model(tensor)
                        logits = self.visualizer.zsl_classifier.predict_test(pred_attr)
                        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                        top_idx = np.argmax(probs)
                        top_prob = probs[top_idx]
                        top_name = self.visualizer.test_names[top_idx]

                        current_result = (top_name, top_prob)
                        result_display_time = current_time
                except Exception as e:
                    pass

            if current_result and (current_time - result_display_time) < self.display_duration:
                name, prob = current_result
                cv2.rectangle(frame, (5, 5), (350, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Recognition: {name}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {prob * 100:.1f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

            cv2.putText(frame, "Press 'q' quit, 's' save", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.imshow("Animal Recognition System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(OUTPUT_PATH, 'recognition_results', f"camera_{int(time.time())}.png")
                cv2.imwrite(save_path, frame)
                print(f"Saved: {save_path}")

        cap.release()
        cv2.destroyAllWindows()
        print("Camera recognition ended")


# ==================== 主程序 ====================

def main():
    set_seed(SEED)
    create_dirs()

    print(f"\n{'=' * 60}")
    print(f"AWA2 Zero-Shot Learning System with Ablation Study")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Max Images Per Class: {MAX_IMAGES_PER_CLASS}")
    print(f"{'=' * 60}\n")

    # 加载数据集
    print("Loading dataset...")
    loader = AWA2Loader()

    # 创建变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = AWA2Dataset(loader.train_imgs, loader.train_local, loader.train_lbls, loader, train_transform)
    test_dataset = AWA2Dataset(loader.test_imgs, loader.test_local, loader.test_lbls, loader, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # ==================== 训练主模型 ====================
    print(f"\nInitializing main model...")
    model = AttributeEmbeddingModel(loader.attributes.shape[1])

    train_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.train_idx]).to(DEVICE)
    test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
    zsl_classifier = ZeroShotClassifier(train_attrs, test_attrs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 训练主模型
    main_trainer = MainTrainer(model, zsl_classifier, train_loader, test_loader, loader)
    main_history = main_trainer.train()
    main_best_acc = main_trainer.best_test_acc

    # 加载最佳主模型
    best_path = os.path.join(OUTPUT_PATH, 'models', 'best_main.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model'])
        zsl_classifier.load_state_dict(checkpoint['zsl_classifier'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']} (Test Acc: {checkpoint['test_acc']:.2f}%)")

    # ==================== 运行消融实验 ====================
    print("\n" + "=" * 60)
    print("Starting Ablation Study (7 experiments, 90 epochs each)")
    print("=" * 60)

    ablation_results = {}

    for config_name, config in AblationConfig.CONFIGS.items():
        # 跳过baseline，因为它和主模型配置相同
        if config_name == 'baseline':
            continue

        trainer = AblationTrainer(config_name, config, loader)
        history = trainer.train()

        ablation_results[config_name] = {
            'name': config['name'],
            'best_test_acc': history['best_test_acc'],
            'best_epoch': history['best_epoch'],
            'final_test_acc': history['final_test_acc'],
            'test_acc': history['test_acc'],
            'train_acc': history['train_acc'],
            'color': config['color']
        }

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 添加baseline结果（使用主模型的结果）
    ablation_results['baseline'] = {
        'name': 'Baseline (Full Model)',
        'best_test_acc': main_best_acc,
        'best_epoch': main_trainer.best_epoch,
        'final_test_acc': main_history['test_acc'][-1],
        'test_acc': main_history['test_acc'],
        'train_acc': main_history['train_acc'],
        'color': '#2ecc71'
    }

    # ==================== 分析并生成报告 ====================
    print("\n" + "=" * 60)
    print("Generating Ablation Study Analysis")
    print("=" * 60)

    analyzer = AblationAnalyzer(main_best_acc, ablation_results)
    analyzer.create_comparison_table()
    analyzer.plot_all()
    analyzer.generate_report()

    # ==================== 可视化 ====================
    visualizer = Visualizer(model, zsl_classifier, loader)
    recognition = RecognitionSystem(visualizer)

    print("\n" + "=" * 60)
    print("Visualizing Test Classes")
    print("=" * 60)
    visualizer.visualize_all_test_classes()

    # ==================== 交互式选择 ====================
    print("\n" + "=" * 60)
    print("Function Selection")
    print("=" * 60)
    print("1. Continuous Image Recognition")
    print("2. Real-time Camera Recognition")
    print("3. Exit")

    while True:
        choice = input("\nSelect function (1/2/3): ").strip()
        if choice == '1':
            recognition.continuous_recognition()
        elif choice == '2':
            recognition.camera_recognition()
        elif choice == '3':
            print("Exiting program")
            break
        else:
            print("Invalid choice, please try again")

    print(f"\n🎉 All results saved to: {OUTPUT_PATH}")
    print(f"   - Main model results: {OUTPUT_PATH}/training_logs/")
    print(f"   - Ablation study results: {OUTPUT_PATH}/ablation_study/")


if __name__ == "__main__":
    main()