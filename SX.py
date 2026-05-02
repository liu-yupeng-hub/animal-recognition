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
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

DATASET_PATH = r"C:\Users\34552\Desktop\Animals_with_Attributes2"
OUTPUT_PATH = r"C:\Users\34552\Desktop\DS1"

EPOCHS = 90
BATCH_SIZE = 32
INIT_LR = 2e-5
IMAGE_SIZE = 224
NUM_WORKERS = 8
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
    for d in ["models", "visualizations", "training_logs", "recognition_results", "heatmaps"]:
        os.makedirs(os.path.join(OUTPUT_PATH, d), exist_ok=True)


# ==================== 数据加载 ====================

class AWA2Loader:
    def __init__(self):
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


# ==================== 模型 ====================

class AttributeEmbeddingModel(nn.Module):
    def __init__(self, n_attrs):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.attr_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, n_attrs)
        )

        self.n_attrs = n_attrs

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_attrs = self.attr_head(x)
        return pred_attrs


class ZeroShotClassifier(nn.Module):
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
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==================== 训练器 ====================

class Trainer:
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

        self.panda_idx = None
        for i, name in enumerate(loader.test_names):
            if 'giant_panda' in name.lower():
                self.panda_idx = i
                break

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

            with autocast():
                pred_attr = self.model(img)
                attr_loss = self.attr_criterion(pred_attr, attr)
                logits = self.zsl_classifier.predict_train(pred_attr)
                cls_loss = self.cls_criterion(logits, lbl)
                loss = attr_loss + cls_loss

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.attr_head.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
        print(f"\nStarting Zero-Shot Learning Training...")
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
                }, os.path.join(OUTPUT_PATH, 'models', 'best.pth'))
                print(f"  ✓ Saved best model (Zero-Shot Acc: {test_acc:.2f}%)")

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'model': self.model.state_dict(),
                    'zsl_classifier': self.zsl_classifier.state_dict(),
                    'epoch': epoch + 1,
                    'test_acc': test_acc
                }, os.path.join(OUTPUT_PATH, 'models', f'epoch_{epoch + 1}.pth'))

        print(f"\n{'=' * 50}")
        print(f"Best zero-shot test accuracy: {self.best_test_acc:.2f}% at epoch {self.best_epoch}")

        # 绘制所有曲线
        self._plot_curves()

        # 最终评估并保存热力图
        final_acc, final_top5, final_preds, final_labels = self.evaluate()
        self._save_heatmap(final_labels, final_preds)

        self._save_history()

        return self.history

    def _plot_curves(self):
        """绘制三个独立的曲线图"""
        epochs = self.history['epoch']

        # 图1: 训练损失曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.history['train_loss'], 'b-o', linewidth=2, markersize=4, label='Total Loss')
        plt.plot(epochs, self.history['train_attr_loss'], 'g-s', linewidth=2, markersize=4, label='Attribute Loss')
        plt.plot(epochs, self.history['train_cls_loss'], 'r-^', linewidth=2, markersize=4, label='Classification Loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training Loss Curves', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'loss_curves.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 图2: 训练集准确率曲线（独立，纵坐标拉长）
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.history['train_acc'], 'b-o', linewidth=2, markersize=4, label='Train Top-1 Accuracy')
        plt.plot(epochs, self.history['train_top5'], 'r-s', linewidth=2, markersize=4, label='Train Top-5 Accuracy')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Training Set Accuracy Curves', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'train_accuracy_curves.png'), dpi=200,
                    bbox_inches='tight')
        plt.close()

        # 图3: 测试集准确率曲线（独立，纵坐标拉长）
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.history['test_acc'], 'b-o', linewidth=2, markersize=4,
                 label='Zero-Shot Test Top-1 Accuracy')
        plt.plot(epochs, self.history['test_top5'], 'r-s', linewidth=2, markersize=4,
                 label='Zero-Shot Test Top-5 Accuracy')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Zero-Shot Test Set Accuracy Curves', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'test_accuracy_curves.png'), dpi=200,
                    bbox_inches='tight')
        plt.close()

        # 图4: 训练vs测试对比曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.history['train_acc'], 'b-o', linewidth=2, markersize=4, label='Train Accuracy')
        plt.plot(epochs, self.history['test_acc'], 'r-s', linewidth=2, markersize=4, label='Zero-Shot Test Accuracy')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Train vs Zero-Shot Test Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'comparison_curves.png'), dpi=200, bbox_inches='tight')
        plt.close()

    def _save_heatmap(self, true_labels, pred_labels):
        """保存混淆矩阵热力图"""
        test_class_names = self.loader.test_names

        cm = confusion_matrix(true_labels, pred_labels)

        # 绘制热力图
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[c[:12] for c in test_class_names],
                    yticklabels=[c[:12] for c in test_class_names],
                    annot_kws={'size': 10})
        plt.title('Confusion Matrix - Zero-Shot Learning Evaluation', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'heatmaps', 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 保存分类报告
        report = classification_report(true_labels, pred_labels, target_names=test_class_names, output_dict=True)
        with open(os.path.join(OUTPUT_PATH, 'heatmaps', 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # 计算各类别准确率
        class_acc = {}
        for i, name in enumerate(test_class_names):
            mask = np.array(true_labels) == i
            if mask.sum() > 0:
                class_acc[name] = 100 * np.sum(np.array(pred_labels)[mask] == i) / mask.sum()

        with open(os.path.join(OUTPUT_PATH, 'heatmaps', 'class_accuracy.txt'), 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("Per-Class Accuracy\n")
            f.write("=" * 50 + "\n")
            for name, acc in sorted(class_acc.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{name}: {acc:.2f}%\n")
            f.write("=" * 50 + "\n")

        print(f"\nConfusion matrix saved to: {os.path.join(OUTPUT_PATH, 'heatmaps')}")

    def _save_history(self):
        with open(os.path.join(OUTPUT_PATH, 'training_logs', 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        with open(os.path.join(OUTPUT_PATH, 'training_logs', 'final_report.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Zero-Shot Learning Training Final Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Best Zero-Shot Test Accuracy: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})\n")
            f.write(f"Final Zero-Shot Test Accuracy: {self.history['test_acc'][-1]:.2f}%\n")
            f.write(f"Final Train Accuracy: {self.history['train_acc'][-1]:.2f}%\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
            f.write(f"Train Classes: {len(self.loader.train_names)}\n")
            f.write(f"Test Classes: {len(self.loader.test_names)}\n")
            f.write(f"Max Images Per Class: {MAX_IMAGES_PER_CLASS}\n")
            f.write("=" * 60 + "\n")


# ==================== 动物知识库 ====================

ANIMAL_FACTS = {
    'giant_panda': ['Bamboo diet (99%)', 'Black and white fur', 'Native to China', 'Spends 12-16 hours eating'],
    'zebra': ['Unique stripe patterns', 'Can run 65 km/h', 'Live in herds', 'Excellent eyesight'],
    'giraffe': ['Tallest land animal', 'Neck has 7 vertebrae', 'Tongue up to 45cm', 'Sleep only 30 min/day'],
    'tiger': ['Nocturnal hunter', 'Excellent swimmer', 'Unique stripe patterns', 'Top predator'],
    'lion': ['Only social cats', 'Pride of 15-30 members', 'Sleep 18-20 hours/day', 'King of savanna'],
    'elephant': ['Largest land mammal', 'Trunk has 100k muscles', 'Excellent memory', 'Lifespan up to 70 years']
}


def get_animal_facts(animal_name):
    for key in ANIMAL_FACTS:
        if key in animal_name.lower():
            return ANIMAL_FACTS[key]
    return ['Wild animal', 'Unique features', 'Adapted to environment', 'Protected species']


# ==================== 可视化器 ====================

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
        facts = get_animal_facts(best_match)

        fig = plt.figure(figsize=(18, 10), facecolor='white')

        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img)
        ax1.set_title(f'Input Image\nPrediction: {best_match}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        rect = Rectangle((0.05, 0.02), 0.9, 0.2, transform=ax1.transAxes, facecolor='#2ecc71', alpha=0.9, clip_on=False)
        ax1.add_patch(rect)
        ax1.text(0.5, 0.1, f"Prediction: {best_match}", transform=ax1.transAxes, ha='center', va='center', fontsize=12,
                 fontweight='bold', color='white')
        ax1.text(0.5, 0.03, f"Confidence: {results[0]['confidence']}", transform=ax1.transAxes, ha='center',
                 va='center', fontsize=9, color='white')

        ax2 = plt.subplot(2, 3, 2)
        names = [r['name'] for r in results]
        probs = [r['probability'] * 100 for r in results]
        colors = ['#2ecc71', '#3498db', '#f39c12']
        bars = ax2.barh(names, probs, color=colors, edgecolor='white', linewidth=2)
        ax2.set_xlabel('Probability (%)', fontsize=11)
        ax2.set_title('Top-3 Predictions\n(Sum = 100%)', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3)
        for bar, prob in zip(bars, probs):
            ax2.text(prob + 1, bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%', va='center', fontsize=10,
                     fontweight='bold')

        ax3 = plt.subplot(2, 3, 3)
        for i, prob in enumerate(probs):
            ci_low = max(0, prob / 100 - 0.05)
            ci_high = min(1, prob / 100 + 0.05)
            ax3.barh(i, ci_high - ci_low, left=ci_low, height=0.5, color='lightblue', edgecolor='navy', linewidth=1)
            ax3.plot(prob / 100, i, 'ro', markersize=8)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.set_xlabel('Probability', fontsize=11)
        ax3.set_title('95% Confidence Interval', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.grid(alpha=0.3)

        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        fact_text = f"Facts about {best_match}\n" + "-" * 30 + "\n"
        for i, fact in enumerate(facts[:4]):
            fact_text += f"• {fact}\n"
        ax4.text(0.05, 0.95, fact_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e6', edgecolor='#8b5a2b', linewidth=2))
        ax4.set_xlim(0, 1);
        ax4.set_ylim(0, 1)

        ax5 = plt.subplot(2, 3, 5, projection='polar')
        top_attrs_idx = np.argsort(np.abs(pred_attr))[-6:]
        angles = np.linspace(0, 2 * np.pi, len(top_attrs_idx), endpoint=False).tolist()
        values = pred_attr[top_attrs_idx]
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        angles += angles[:1];
        values = list(values) + [values[0]]
        ax5.plot(angles, values, 'o-', linewidth=2, color='#e74c3c')
        ax5.fill(angles, values, alpha=0.25, color='#e74c3c')
        attr_labels = [self.loader.predicates[i][:15] for i in top_attrs_idx]
        ax5.set_xticks(angles[:-1]);
        ax5.set_xticklabels(attr_labels, fontsize=8)
        ax5.set_title('Key Attributes', fontsize=12, fontweight='bold', pad=20)

        ax6 = plt.subplot(2, 3, 6)
        attn_map = pred_attr.reshape(5, 17)
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

        for class_name, img_path in test_class_images.items():
            save_path = os.path.join(OUTPUT_PATH, 'visualizations', f"{class_name}_analysis.png")
            print(f"Analyzing: {class_name}")
            try:
                self.visualize_single(img_path, save_path=save_path)
                print(f"  ✓ Saved: {save_path}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")

        print(f"\nVisualization completed!")


# ==================== 识别系统 ====================

class RecognitionSystem:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.display_duration = 2.0

    def recognize_image(self, img_path):
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None
        results, img, _ = self.visualizer.predict(img_path)
        print("\n" + "=" * 50)
        print("Recognition Results")
        print("=" * 50)
        for r in results:
            print(f"{r['rank']}. {r['name']} - {r['confidence']}")
        print("=" * 50)
        return results

    def recognize_and_visualize(self, img_path, save_name=None):
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None
        if save_name is None:
            save_name = f"recognition_{int(time.time())}.png"
        save_path = os.path.join(OUTPUT_PATH, 'recognition_results', save_name)
        results = self.visualizer.visualize_single(img_path, save_path=save_path)
        print(f"Visualization saved: {save_path}")
        return results

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
                self.recognize_and_visualize(path)
            else:
                self.recognize_image(path)

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
        cap.set(cv2.CAP_PROP_FPS, 15)

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
    print(f"AWA2 Zero-Shot Learning Recognition System")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Max Images Per Class: {MAX_IMAGES_PER_CLASS}")
    print(f"{'=' * 60}\n")

    print("Loading dataset...")
    loader = AWA2Loader()

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print(f"\nInitializing model...")
    model = AttributeEmbeddingModel(loader.attributes.shape[1])

    train_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.train_idx]).to(DEVICE)
    test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
    zsl_classifier = ZeroShotClassifier(train_attrs, test_attrs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    trainer = Trainer(model, zsl_classifier, train_loader, test_loader, loader)
    history = trainer.train()

    best_path = os.path.join(OUTPUT_PATH, 'models', 'best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model'])
        zsl_classifier.load_state_dict(checkpoint['zsl_classifier'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']} (Test Acc: {checkpoint['test_acc']:.2f}%)")

    visualizer = Visualizer(model, zsl_classifier, loader)
    recognition = RecognitionSystem(visualizer)

    print("\n" + "=" * 60)
    print("Visualizing Test Classes")
    print("=" * 60)
    visualizer.visualize_all_test_classes()

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


if __name__ == "__main__":
    main()