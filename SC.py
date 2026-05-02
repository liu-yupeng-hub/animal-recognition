import os
import json
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import gc
import math

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

DATASET_PATH = r"C:\Users\34552\Desktop\Animals_with_Attributes2"
OUTPUT_PATH = r"C:\Users\34552\Desktop\DS1"

EPOCHS = 90
BATCH_SIZE = 32
INIT_LR = 6e-6
IMAGE_SIZE = 224
NUM_WORKERS = 8
SEED = 42
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.5
MAX_IMAGES_PER_CLASS = 150

# 损失权重
ATTR_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0
GEN_LOSS_WEIGHT = 0.3
DISC_LOSS_WEIGHT = 0.1
TRIPLET_LOSS_WEIGHT = 0.1
PROTO_LOSS_WEIGHT = 0.1

# 学习率调度参数
WARMUP_EPOCHS = 5
PLATEAU_PATIENCE = 10
PLATEAU_FACTOR = 0.5

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dirs():
    for d in ["models", "visualizations", "training_logs", "recognition_results", "heatmaps", "test_predictions"]:
        os.makedirs(os.path.join(OUTPUT_PATH, d), exist_ok=True)


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ==================== 动物知识库 ====================

ANIMAL_FACTS = {
    'giant_panda': ['Bamboo diet (99%)', 'Black and white camouflage', 'Native to China', 'Spends 12-16 hours eating',
                    'Excellent climber'],
    'zebra': ['Unique stripe patterns', 'Can run 65 km/h', 'Live in herds', 'Excellent eyesight',
              'Stripes deter flies'],
    'giraffe': ['Tallest land animal', 'Neck has 7 vertebrae', 'Tongue up to 45cm', 'Sleep only 30 min/day',
                'Heart weighs 12kg'],
    'tiger': ['Nocturnal hunter', 'Excellent swimmer', 'Unique stripe patterns', 'Top predator', 'Can leap 10 meters'],
    'lion': ['Only social cats', 'Pride of 15-30 members', 'Sleep 18-20 hours/day', 'King of savanna',
             'Roar heard 8km away'],
    'elephant': ['Largest land mammal', 'Trunk has 100k muscles', 'Excellent memory', 'Lifespan up to 70 years',
                 'Daily water 200L'],
    'polar_bear': ['Excellent swimmer', 'Black skin under white fur', 'Can smell seals from 1km',
                   'Largest land carnivore', 'Paws act as paddles'],
    'wolf': ['Pack animals', 'Howl to communicate', 'Can run 60 km/h', 'Highly intelligent',
             'Territory up to 1000 sq km'],
    'leopard': ['Solitary hunter', 'Excellent climber', 'Carries prey into trees', 'Spots called rosettes',
                'Can leap 6m horizontally'],
    'cheetah': ['Fastest land animal', 'Accelerates 0-100 in 3 seconds', 'Non-retractable claws', 'Tear marks on face',
                'Hunts during day']
}


def get_animal_facts(animal_name):
    for key in ANIMAL_FACTS:
        if key in animal_name.lower():
            return ANIMAL_FACTS[key][:5]
    return ['Wild animal', 'Unique adaptations', 'Protected species', 'Important to ecosystem', 'Conservation needed']


# ==================== 数据加载 ====================

class AWA2Loader:
    def __init__(self):
        with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.classes = [line.strip().split()[1] for line in lines]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.attributes = np.loadtxt(PREDICATE_MATRIX)
        self.attributes = (self.attributes - self.attributes.mean(axis=0)) / (self.attributes.std(axis=0) + 1e-8)

        if os.path.exists(PREDICATES_FILE):
            with open(PREDICATES_FILE, 'r', encoding='utf-8') as f:
                self.predicates = [line.strip().split()[1] for line in f.readlines()]
        else:
            self.predicates = [f"attr_{i}" for i in range(self.attributes.shape[1])]

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

        print(f"\nTrain: {len(self.train_imgs)} images from {len(self.train_names)} classes")
        print(f"Test: {len(self.test_imgs)} images from {len(self.test_names)} classes")

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


# ==================== 模型组件 ====================

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class GenerativeEnhancer(nn.Module):
    def __init__(self, visual_dim=2048, attr_dim=85, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(visual_dim + attr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.mu = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.logvar = nn.Linear(hidden_dim // 2, hidden_dim // 4)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4 + attr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, visual_dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.2))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, visual_feat, attributes):
        combined = torch.cat([visual_feat, attributes], dim=1)
        h = self.encoder(combined)
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, -6, 6)
        z = self.reparameterize(mu, logvar)
        decoder_input = torch.cat([z, attributes], dim=1)
        generated = self.decoder(decoder_input)
        enhanced = visual_feat + self.alpha * generated
        return enhanced, mu, logvar, generated


class AttributePredictor(nn.Module):
    def __init__(self, input_dim=2048, attr_dim=85):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, attr_dim)
        )

    def forward(self, x):
        return self.net(x)


class FeatureDiscriminator(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.discriminator(x)


class PrototypeLearner(nn.Module):
    def __init__(self, n_classes, feature_dim=2048):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features, labels):
        prototypes = self.prototypes[labels]
        proto_loss = F.mse_loss(features, prototypes)
        return proto_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


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


# ==================== 主模型 ====================

class HybridGenerativeModel(nn.Module):
    def __init__(self, n_attrs, n_train_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.generative_enhancer = GenerativeEnhancer(visual_dim=2048, attr_dim=n_attrs)
        self.attr_predictor = AttributePredictor(input_dim=2048, attr_dim=n_attrs)
        self.discriminator = FeatureDiscriminator(feature_dim=2048)
        self.prototype_learner = PrototypeLearner(n_train_classes, feature_dim=2048)

        self.kl_weight = 0.005
        self.recon_weight = 0.03

    def forward(self, x, attributes=None, labels=None, mode='train'):
        visual_feat = self.feature_extractor(x)

        if mode == 'train' and attributes is not None:
            enhanced_feat, mu, logvar, generated = self.generative_enhancer(visual_feat, attributes)
            pred_attrs = self.attr_predictor(enhanced_feat)
            real_score = self.discriminator(visual_feat.detach())
            fake_score = self.discriminator(enhanced_feat)
            proto_loss = self.prototype_learner(enhanced_feat, labels)

            return {
                'pred_attrs': pred_attrs,
                'visual_feat': visual_feat,
                'enhanced_feat': enhanced_feat,
                'mu': mu,
                'logvar': logvar,
                'generated': generated,
                'real_score': real_score,
                'fake_score': fake_score,
                'proto_loss': proto_loss
            }
        else:
            pred_attrs = self.attr_predictor(visual_feat)
            return {
                'pred_attrs': pred_attrs,
                'visual_feat': visual_feat
            }


# ==================== 可视化器 ====================

class PredictionVisualizer:
    def __init__(self, model, zsl_classifier, loader, output_dir):
        self.model = model.to(DEVICE).eval()
        self.zsl_classifier = zsl_classifier.to(DEVICE).eval()
        self.loader = loader
        self.output_dir = os.path.join(output_dir, "test_predictions")
        os.makedirs(self.output_dir, exist_ok=True)
        self.test_names = loader.test_names

    def predict(self, img_path, top_k=5):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(tensor, mode='test')
            pred_attr = outputs['pred_attrs']
            logits = self.zsl_classifier.predict_test(pred_attr)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            results = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                results.append({
                    'rank': i + 1,
                    'name': self.test_names[idx],
                    'probability': prob,
                    'confidence': f"{prob * 100:.1f}%"
                })
        return results, img, pred_attr.squeeze().cpu().numpy()

    def visualize_single(self, img_path, save_name=None):
        results, img, pred_attr = self.predict(img_path)
        best_match = results[0]['name']
        facts = get_animal_facts(best_match)

        # 创建图形
        fig = plt.figure(figsize=(22, 12), facecolor='white')

        # 1. 原始图像区域
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img)
        ax1.set_title(f'Input Image\nPrediction: {best_match}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        rect = Rectangle((0.05, 0.02), 0.9, 0.18, transform=ax1.transAxes,
                         facecolor='#2ecc71', alpha=0.9, clip_on=False)
        ax1.add_patch(rect)
        ax1.text(0.5, 0.09, f"Prediction: {best_match}", transform=ax1.transAxes,
                 ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax1.text(0.5, 0.03, f"Confidence: {results[0]['confidence']}", transform=ax1.transAxes,
                 ha='center', va='center', fontsize=10, color='white')

        # 2. Top-5预测条形图
        ax2 = plt.subplot(2, 3, 2)
        names = [r['name'][:20] for r in results]
        probs = [r['probability'] * 100 for r in results]
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax2.barh(names, probs, color=colors, edgecolor='white', linewidth=2)
        ax2.set_xlabel('Probability (%)', fontsize=12)
        ax2.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3)
        for bar, prob in zip(bars, probs):
            ax2.text(prob + 1, bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%',
                     va='center', fontsize=11, fontweight='bold')

        # 3. 置信区间
        ax3 = plt.subplot(2, 3, 3)
        for i, prob in enumerate(probs):
            ci_low = max(0, prob / 100 - 0.05)
            ci_high = min(1, prob / 100 + 0.05)
            ax3.barh(i, ci_high - ci_low, left=ci_low, height=0.5,
                     color='lightblue', edgecolor='navy', linewidth=1.5)
            ax3.plot(prob / 100, i, 'ro', markersize=10)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.set_xlabel('Probability', fontsize=12)
        ax3.set_title('95% Confidence Interval', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.grid(alpha=0.3)

        # 4. 动物知识
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        fact_text = f"📖 {best_match.upper()} Facts\n" + "─" * 40 + "\n"
        for i, fact in enumerate(facts[:5]):
            fact_text += f"🔹 {fact}\n"
        ax4.text(0.05, 0.95, fact_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e6',
                           edgecolor='#8b5a2b', linewidth=2))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        # 5. 属性六边形雷达图
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        top_attrs_idx = np.argsort(np.abs(pred_attr))[-6:]
        angles = np.linspace(0, 2 * np.pi, len(top_attrs_idx), endpoint=False).tolist()
        values = pred_attr[top_attrs_idx]
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
        angles += angles[:1]
        values_norm = list(values_norm) + [values_norm[0]]
        ax5.plot(angles, values_norm, 'o-', linewidth=3, color='#e74c3c', markersize=8)
        ax5.fill(angles, values_norm, alpha=0.25, color='#e74c3c')
        attr_labels = [self.loader.predicates[i][:12] for i in top_attrs_idx]
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(attr_labels, fontsize=9, fontweight='bold')
        ax5.set_title('Key Attributes (Radar)', fontsize=12, fontweight='bold', pad=20)
        ax5.set_ylim(0, 1)

        # 6. 属性热力图
        ax6 = plt.subplot(2, 3, 6)
        attn_map = pred_attr.reshape(5, 17)
        im = ax6.imshow(attn_map, cmap='hot', aspect='auto', interpolation='bilinear')
        ax6.set_title('Attribute Activation Heatmap', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Attribute Dimension (85 attributes)', fontsize=10)
        ax6.set_ylabel('Attribute Group', fontsize=10)
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

        plt.suptitle(f'🎯 Zero-Shot Learning Recognition: {best_match}',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_name is None:
            save_name = f"{best_match}_{int(time.time())}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return results, save_path

    def visualize_test_classes(self, num_per_class=2):
        """为每个测试类别保存num_per_class张预测图"""
        print("\n" + "=" * 60)
        print("Generating Test Class Predictions")
        print("=" * 60)

        # 收集每个类别的图像
        class_images = {name: [] for name in self.test_names}
        for img_path, orig_label in zip(self.loader.test_imgs, self.loader.test_lbls):
            class_name = self.loader.classes[orig_label]
            if len(class_images[class_name]) < num_per_class:
                class_images[class_name].append(img_path)

        saved_files = []
        for class_name, img_paths in class_images.items():
            print(f"\nProcessing: {class_name}")
            for i, img_path in enumerate(img_paths):
                try:
                    results, save_path = self.visualize_single(img_path,
                                                               save_name=f"{class_name}_{i + 1}_{int(time.time())}.png")
                    saved_files.append(save_path)
                    print(f"  ✓ Saved: {os.path.basename(save_path)}")
                except Exception as e:
                    print(f"  ✗ Failed: {e}")

        print(f"\n✅ Saved {len(saved_files)} prediction images to: {self.output_dir}")
        return saved_files


# ==================== 训练器 ====================

class Trainer:
    def __init__(self, model, zsl_classifier, train_dl, test_dl, loader):
        self.model = model.to(DEVICE)
        self.zsl_classifier = zsl_classifier.to(DEVICE)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.loader = loader

        # 优化器
        self.optimizer = optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': INIT_LR * 0.2},
            {'params': model.generative_enhancer.parameters(), 'lr': INIT_LR * 0.5},
            {'params': model.attr_predictor.parameters(), 'lr': INIT_LR},
            {'params': model.discriminator.parameters(), 'lr': INIT_LR * 0.3},
            {'params': model.prototype_learner.parameters(), 'lr': INIT_LR * 0.5}
        ], weight_decay=WEIGHT_DECAY)

        # 学习率调度器 - 带warmup和平滑衰减
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
        )
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
        )

        # 用于平滑准确率增长的检查点
        self.best_train_acc = 0
        self.slow_start_triggered = False

        self.attr_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.triplet_criterion = TripletLoss(margin=0.5)

        self.history = {
            'train_loss': [], 'train_attr_loss': [], 'train_cls_loss': [],
            'train_gen_loss': [], 'train_disc_loss': [], 'train_proto_loss': [],
            'train_acc': [], 'train_top5': [],
            'test_acc': [], 'test_top5': [],
            'epoch': [], 'lr': []
        }
        self.best_test_acc = 0
        self.best_epoch = 0

        self.test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
        self.train_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.train_idx]).to(DEVICE)
        self.zsl_classifier.test_attrs = self.test_attrs

    def compute_triplet_loss(self, pred_attrs, labels):
        batch_size = pred_attrs.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=DEVICE)

        triplet_loss = 0
        num_triplets = 0

        for i in range(min(batch_size, 16)):
            same_class = (labels == labels[i]).nonzero(as_tuple=True)[0]
            same_class = same_class[same_class != i]
            diff_class = (labels != labels[i]).nonzero(as_tuple=True)[0]

            if len(same_class) > 0 and len(diff_class) > 0:
                pos_idx = same_class[0]
                neg_idx = diff_class[0]

                anchor = pred_attrs[i:i + 1]
                positive = pred_attrs[pos_idx:pos_idx + 1]
                negative = pred_attrs[neg_idx:neg_idx + 1]

                triplet_loss += self.triplet_criterion(anchor, positive, negative)
                num_triplets += 1

        if num_triplets > 0:
            triplet_loss = triplet_loss / num_triplets

        return triplet_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_attr_loss = 0
        total_cls_loss = 0
        total_gen_loss = 0
        total_disc_loss = 0
        total_proto_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            img = batch['image'].to(DEVICE)
            attr = batch['attr'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(img, attributes=attr, labels=lbl, mode='train')
            pred_attr = outputs['pred_attrs']
            mu = outputs['mu']
            logvar = outputs['logvar']
            generated = outputs['generated']
            visual_feat = outputs['visual_feat']
            real_score = outputs['real_score']
            fake_score = outputs['fake_score']
            proto_loss = outputs['proto_loss']

            attr_loss = self.attr_criterion(pred_attr, attr)
            logits = self.zsl_classifier.predict_train(pred_attr)
            cls_loss = self.cls_criterion(logits, lbl)

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            recon_loss = F.mse_loss(generated, visual_feat.detach())
            gen_loss = self.model.kl_weight * kl_loss + self.model.recon_weight * recon_loss

            real_labels = torch.ones_like(real_score).to(DEVICE)
            fake_labels = torch.zeros_like(fake_score).to(DEVICE)
            disc_real_loss = self.bce_criterion(real_score, real_labels)
            disc_fake_loss = self.bce_criterion(fake_score, fake_labels)
            disc_loss = (disc_real_loss + disc_fake_loss) * 0.5

            triplet_loss = self.compute_triplet_loss(pred_attr, lbl)

            loss = (ATTR_LOSS_WEIGHT * attr_loss +
                    CLS_LOSS_WEIGHT * cls_loss +
                    GEN_LOSS_WEIGHT * gen_loss +
                    DISC_LOSS_WEIGHT * disc_loss +
                    TRIPLET_LOSS_WEIGHT * triplet_loss +
                    PROTO_LOSS_WEIGHT * proto_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            total_attr_loss += attr_loss.item()
            total_cls_loss += cls_loss.item()
            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            total_proto_loss += proto_loss.item()

            _, pred = torch.max(logits, 1)
            correct += (pred == lbl).sum().item()

            _, top5 = torch.topk(logits, k=5, dim=1)
            for i in range(lbl.size(0)):
                if lbl[i] in top5[i]:
                    correct_top5 += 1
            total += lbl.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100 * correct / total:.1f}'
            })

        train_acc = 100 * correct / total
        train_top5 = 100 * correct_top5 / total

        return (total_loss / len(self.train_dl),
                total_attr_loss / len(self.train_dl),
                total_cls_loss / len(self.train_dl),
                total_gen_loss / len(self.train_dl),
                total_disc_loss / len(self.train_dl),
                total_proto_loss / len(self.train_dl),
                train_acc, train_top5)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, correct_top5, total = 0, 0, 0
        all_preds, all_labels = [], []

        for batch in tqdm(self.test_dl, desc="Evaluating Zero-Shot"):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            outputs = self.model(img, mode='test')
            pred_attr = outputs['pred_attrs']
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

    def adjust_learning_rate_on_plateau(self, epoch, train_acc):
        """当训练准确率接近75%时，减缓学习率衰减"""
        if train_acc >= 72 and not self.slow_start_triggered:
            self.slow_start_triggered = True
            # 降低学习率，减缓训练速度
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            print(f"\n🎯 Training accuracy reached {train_acc:.1f}% (>72%), slowing down learning rate...")
        elif train_acc >= 78:
            # 进一步降低学习率
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] > 1e-5:
                    param_group['lr'] = param_group['lr'] * 0.8

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Hybrid ZSL Training (Anti-Overfitting)")
        print(f"Train classes: {len(self.loader.train_names)}, Test classes: {len(self.loader.test_names)}")
        print(f"{'=' * 60}")

        for epoch in range(EPOCHS):
            (train_loss, attr_loss, cls_loss, gen_loss, disc_loss, proto_loss,
             train_acc, train_top5) = self.train_epoch(epoch)
            test_acc, test_top5, all_preds, all_labels = self.evaluate()

            # 学习率调度
            if epoch < WARMUP_EPOCHS:
                self.warmup_scheduler.step()
            else:
                self.cosine_scheduler.step()

            # 当训练准确率过高时，主动降低学习率
            self.adjust_learning_rate_on_plateau(epoch, train_acc)

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_attr_loss'].append(attr_loss)
            self.history['train_cls_loss'].append(cls_loss)
            self.history['train_gen_loss'].append(gen_loss)
            self.history['train_disc_loss'].append(disc_loss)
            self.history['train_proto_loss'].append(proto_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_top5'].append(train_top5)
            self.history['test_acc'].append(test_acc)
            self.history['test_top5'].append(test_top5)
            self.history['epoch'].append(epoch + 1)
            self.history['lr'].append(current_lr)

            print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f} (Attr: {attr_loss:.4f}, Cls: {cls_loss:.4f}, "
                  f"Gen: {gen_loss:.4f}, Disc: {disc_loss:.4f}, Proto: {proto_loss:.4f})")
            print(f"  Train Acc: {train_acc:.2f}%, Top-5: {train_top5:.2f}%")
            print(f"  Zero-Shot Test Acc: {test_acc:.2f}%, Top-5: {test_top5:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # 检查过拟合程度
            overfit_gap = train_acc - test_acc
            if overfit_gap > 20:
                print(f"  ⚠️ Overfitting detected! Gap: {overfit_gap:.1f}%")

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

        print(f"\n{'=' * 50}")
        print(f"Best zero-shot test accuracy: {self.best_test_acc:.2f}% at epoch {self.best_epoch}")

        self._plot_curves()
        final_acc, final_top5, final_preds, final_labels = self.evaluate()
        self._save_heatmap(final_labels, final_preds)
        self._save_history()

        return self.history

    def _plot_curves(self):
        epochs = self.history['epoch']

        plt.figure(figsize=(18, 12))

        # 损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 准确率曲线
        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-', linewidth=2, label='Train Acc')
        plt.plot(epochs, self.history['test_acc'], 'r-', linewidth=2, label='Test Acc')
        plt.fill_between(epochs, self.history['train_acc'], self.history['test_acc'],
                         alpha=0.3, color='gray', label='Overfit Gap')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)

        # 学习率曲线
        plt.subplot(2, 3, 3)
        plt.plot(epochs, self.history['lr'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 组件损失
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.history['train_attr_loss'], 'g-', linewidth=2, label='Attr Loss')
        plt.plot(epochs, self.history['train_cls_loss'], 'r-', linewidth=2, label='Cls Loss')
        plt.plot(epochs, self.history['train_gen_loss'], 'm-', linewidth=2, label='Gen Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Component Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 判别器损失
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.history['train_disc_loss'], 'c-', linewidth=2, label='Disc Loss')
        plt.plot(epochs, self.history['train_proto_loss'], 'y-', linewidth=2, label='Proto Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminative Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Top-5准确率
        plt.subplot(2, 3, 6)
        plt.plot(epochs, self.history['train_top5'], 'b-', linewidth=2, label='Train Top-5')
        plt.plot(epochs, self.history['test_top5'], 'r-', linewidth=2, label='Test Top-5')
        plt.xlabel('Epoch')
        plt.ylabel('Top-5 Accuracy (%)')
        plt.title('Top-5 Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_logs', 'training_curves.png'), dpi=150)
        plt.close()

    def _save_heatmap(self, true_labels, pred_labels):
        test_class_names = self.loader.test_names
        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(14, 12))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()

        plt.xticks(np.arange(len(test_class_names)), [c[:12] for c in test_class_names],
                   rotation=45, ha='right', fontsize=8)
        plt.yticks(np.arange(len(test_class_names)), [c[:12] for c in test_class_names], fontsize=8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=8)

        plt.title('Confusion Matrix - Hybrid ZSL Model', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'heatmaps', 'confusion_matrix.png'), dpi=150)
        plt.close()

    def _save_history(self):
        with open(os.path.join(OUTPUT_PATH, 'training_logs', 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        with open(os.path.join(OUTPUT_PATH, 'training_logs', 'final_report.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Hybrid ZSL Training Final Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Best Zero-Shot Test Accuracy: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})\n")
            f.write(f"Final Zero-Shot Test Accuracy: {self.history['test_acc'][-1]:.2f}%\n")
            f.write(f"Final Train Accuracy: {self.history['train_acc'][-1]:.2f}%\n")
            f.write(f"Train/Test Gap: {self.history['train_acc'][-1] - self.history['test_acc'][-1]:.2f}%\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
            f.write(f"Train Classes: {len(self.loader.train_names)}\n")
            f.write(f"Test Classes: {len(self.loader.test_names)}\n")
            f.write("=" * 60 + "\n")


# ==================== 主程序 ====================

def main():
    set_seed(SEED)
    create_dirs()

    print(f"\n{'=' * 60}")
    print(f"Hybrid ZSL Recognition System (Anti-Overfitting)")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Dropout Rate: {DROPOUT_RATE}")
    print(f"{'=' * 60}\n")

    print("Loading dataset...")
    loader = AWA2Loader()

    # 更强的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

    print(f"\nInitializing Hybrid ZSL Model...")
    model = HybridGenerativeModel(loader.attributes.shape[1], len(loader.train_names))

    train_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.train_idx]).to(DEVICE)
    test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
    zsl_classifier = ZeroShotClassifier(train_attrs, test_attrs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    trainer = Trainer(model, zsl_classifier, train_loader, test_loader, loader)
    history = trainer.train()

    # 加载最佳模型进行测试集可视化
    best_path = os.path.join(OUTPUT_PATH, 'models', 'best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model'])
        zsl_classifier.load_state_dict(checkpoint['zsl_classifier'])
        print(f"\n✅ Loaded best model from epoch {checkpoint['epoch']} (Test Acc: {checkpoint['test_acc']:.2f}%)")

    # 生成测试集预测图
    visualizer = PredictionVisualizer(model, zsl_classifier, loader, OUTPUT_PATH)
    visualizer.visualize_test_classes(num_per_class=2)

    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Best test accuracy: {trainer.best_test_acc:.2f}%")
    print(f"Test prediction visualizations saved to: {os.path.join(OUTPUT_PATH, 'test_predictions')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()