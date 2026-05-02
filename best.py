import os
import json
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

DATASET_PATH = r"C:\Users\34552\Desktop\Animals_with_Attributes2"
MODEL_PATH = r"C:\Users\34552\Desktop\DS -基于属性-属性热力激活图\models\best.pth"
OUTPUT_PATH = r"C:\Users\34552\Desktop\best"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 动物知识库 ====================

ANIMAL_FACTS = {
    'giant_panda': ['Bamboo diet (99%)', 'Black and white fur', 'Native to China', 'Spends 12-16 hours eating'],
    'zebra': ['Unique stripe patterns', 'Can run 65 km/h', 'Live in herds', 'Excellent eyesight'],
    'giraffe': ['Tallest land animal', 'Neck has 7 vertebrae', 'Tongue up to 45cm', 'Sleep only 30 min/day'],
    'tiger': ['Nocturnal hunter', 'Excellent swimmer', 'Unique stripe patterns', 'Top predator'],
    'lion': ['Only social cats', 'Pride of 15-30 members', 'Sleep 18-20 hours/day', 'King of savanna'],
    'elephant': ['Largest land mammal', 'Trunk has 100k muscles', 'Excellent memory', 'Lifespan up to 70 years'],
    'polar_bear': ['Excellent swimmer', 'Thick blubber layer', 'Black skin under fur', 'Can smell prey from 1km'],
    'wolf': ['Pack hunters', 'Howl to communicate', 'Excellent stamina', 'Top of food chain'],
    'chimpanzee': ['Use tools', 'Complex social structure', 'Share 98% DNA with humans', 'Learn sign language'],
    'dolphin': ['Highly intelligent', 'Use echolocation', 'Sleep with one eye open', 'Complex communication']
}


def get_animal_facts(animal_name):
    for key in ANIMAL_FACTS:
        if key in animal_name.lower():
            return ANIMAL_FACTS[key]
    return ['Wild animal', 'Unique features', 'Adapted to environment', 'Protected species']


# ==================== 数据加载器 ====================

class AWA2Loader:
    def __init__(self):
        classes_file = os.path.join(DATASET_PATH, "classes.txt")
        predicate_matrix = os.path.join(DATASET_PATH, "predicate-matrix-continuous.txt")
        predicates_file = os.path.join(DATASET_PATH, "predicates.txt")
        test_classes = os.path.join(DATASET_PATH, "testclasses.txt")

        with open(classes_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.classes = [line.strip().split()[1] for line in lines]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.attributes = np.loadtxt(predicate_matrix)

        if os.path.exists(predicates_file):
            with open(predicates_file, 'r', encoding='utf-8') as f:
                self.predicates = [line.strip().split()[1] for line in f.readlines()]
        else:
            self.predicates = [f"attr_{i}" for i in range(self.attributes.shape[1])]

        # 标准化属性
        self.attributes = (self.attributes - self.attributes.mean(axis=0)) / (self.attributes.std(axis=0) + 1e-8)

        with open(test_classes, 'r', encoding='utf-8') as f:
            self.test_names = [line.strip() for line in f.readlines()]

        self.test_idx = [self.class_to_idx[c] for c in self.test_names]
        self.test_map = {o: i for i, o in enumerate(self.test_idx)}


# ==================== 模型定义 ====================

class AttributeEmbeddingModel(nn.Module):
    def __init__(self, n_attrs, dropout_rate=0.4):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 冻结ResNet参数
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
    def __init__(self, test_attrs):
        super().__init__()
        self.register_buffer('test_attrs', test_attrs)

    def predict_test(self, pred_attrs):
        pred_norm = F.normalize(pred_attrs, p=2, dim=1)
        test_norm = F.normalize(self.test_attrs, p=2, dim=1)
        return torch.mm(pred_norm, test_norm.t()) / 0.1


# ==================== 可视化器 ====================

class Visualizer:
    def __init__(self, model, zsl_classifier, loader):
        self.model = model.to(DEVICE).eval()
        self.zsl_classifier = zsl_classifier.to(DEVICE).eval()
        self.loader = loader
        self.test_names = loader.test_names

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_path, top_k=3):
        """预测图像"""
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)

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
                    'confidence': f"{prob * 100:.1f}%",
                    'raw_prob': probs[idx]
                })

        return results, img, pred_attr.squeeze().cpu().numpy()

    def visualize_single(self, img_path, save_path=None):
        """可视化单张图像识别结果"""
        results, img, pred_attr = self.predict(img_path)
        best_match = results[0]['name']
        facts = get_animal_facts(best_match)

        fig = plt.figure(figsize=(18, 10), facecolor='white')

        # 图1: 输入图像和预测结果
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img)
        ax1.set_title(f'Input Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 添加预测标签
        rect = Rectangle((0.05, 0.02), 0.9, 0.2, transform=ax1.transAxes,
                         facecolor='#2ecc71', alpha=0.9, clip_on=False)
        ax1.add_patch(rect)
        ax1.text(0.5, 0.1, f"Prediction: {best_match}", transform=ax1.transAxes,
                 ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax1.text(0.5, 0.03, f"Confidence: {results[0]['confidence']}", transform=ax1.transAxes,
                 ha='center', va='center', fontsize=9, color='white')

        # 图2: Top-3预测
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
            ax2.text(prob + 1, bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%',
                     va='center', fontsize=10, fontweight='bold')

        # 图3: 置信区间
        ax3 = plt.subplot(2, 3, 3)
        raw_probs = [r['raw_prob'] for r in results]
        for i, prob in enumerate(raw_probs):
            ci_low = max(0, prob - 0.05)
            ci_high = min(1, prob + 0.05)
            ax3.barh(i, ci_high - ci_low, left=ci_low, height=0.5,
                     color='lightblue', edgecolor='navy', linewidth=1)
            ax3.plot(prob, i, 'ro', markersize=8)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.set_xlabel('Probability', fontsize=11)
        ax3.set_title('95% Confidence Interval', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.grid(alpha=0.3)

        # 图4: 动物趣闻
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        fact_text = f"Facts about {best_match}\n" + "-" * 30 + "\n"
        for i, fact in enumerate(facts[:4]):
            fact_text += f"• {fact}\n"
        ax4.text(0.05, 0.95, fact_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e6',
                           edgecolor='#8b5a2b', linewidth=2))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        # 图5: 关键属性雷达图
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        top_attrs_idx = np.argsort(np.abs(pred_attr))[-6:]
        angles = np.linspace(0, 2 * np.pi, len(top_attrs_idx), endpoint=False).tolist()
        values = pred_attr[top_attrs_idx]
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        angles += angles[:1]
        values = list(values) + [values[0]]
        ax5.plot(angles, values, 'o-', linewidth=2, color='#e74c3c')
        ax5.fill(angles, values, alpha=0.25, color='#e74c3c')
        if hasattr(self.loader, 'predicates'):
            attr_labels = [self.loader.predicates[i][:12] for i in top_attrs_idx]
        else:
            attr_labels = [f"Attr_{i}" for i in top_attrs_idx]
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(attr_labels, fontsize=8)
        ax5.set_title('Key Attributes', fontsize=12, fontweight='bold', pad=20)

        # 图6: 属性激活图
        ax6 = plt.subplot(2, 3, 6)
        attn_map = pred_attr.reshape(5, -1) if len(pred_attr) >= 85 else pred_attr.reshape(5, 17)
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


# ==================== 识别系统 ====================

class RecognitionSystem:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.display_duration = 2.0

    def recognize_image(self, img_path):
        """识别单张图像"""
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None

        results, _, _ = self.visualizer.predict(img_path)

        print("\n" + "=" * 50)
        print("Recognition Results")
        print("=" * 50)
        for r in results:
            print(f"{r['rank']}. {r['name']} - {r['confidence']}")
        print("=" * 50)

        return results

    def recognize_and_visualize(self, img_path, save_name=None):
        """识别并可视化"""
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None

        if save_name is None:
            save_name = f"recognition_{int(time.time())}.png"

        save_path = os.path.join(OUTPUT_PATH, 'recognition_results', save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        results = self.visualizer.visualize_single(img_path, save_path=save_path)
        print(f"Visualization saved: {save_path}")

        return results

    def continuous_recognition(self):
        """连续识别模式"""
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
        """实时摄像头识别"""
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

            # 定期处理帧
            if frame_count % process_interval == 0:
                try:
                    # 预处理图像
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    tensor = self.visualizer.transform(pil_img).unsqueeze(0).to(DEVICE)

                    # 推理
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

            # 显示识别结果
            if current_result and (current_time - result_display_time) < self.display_duration:
                name, prob = current_result
                # 绘制背景框
                cv2.rectangle(frame, (5, 5), (350, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Recognition: {name}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {prob * 100:.1f}%", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 操作提示
            cv2.putText(frame, "Press 'q' quit, 's' save", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Animal Recognition System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(OUTPUT_PATH, 'recognition_results',
                                         f"camera_{int(time.time())}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, frame)
                print(f"Saved: {save_path}")

        cap.release()
        cv2.destroyAllWindows()
        print("Camera recognition ended")

    def batch_recognition(self, folder_path):
        """批量识别文件夹中的图像"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"No images found in {folder_path}")
            return

        print(f"\nFound {len(image_files)} images")

        results_summary = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            print(f"\nProcessing: {img_file}")

            results, _, _ = self.visualizer.predict(img_path)
            top_result = results[0]

            results_summary.append({
                'file': img_file,
                'prediction': top_result['name'],
                'confidence': top_result['confidence']
            })

            print(f"  Prediction: {top_result['name']} - {top_result['confidence']}")

        # 保存批量识别结果
        summary_path = os.path.join(OUTPUT_PATH, 'recognition_results', 'batch_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        print(f"\nBatch recognition completed! Summary saved to: {summary_path}")

        return results_summary


# ==================== 主程序 ====================

def main():
    print(f"\n{'=' * 60}")
    print(f"AWA2 Zero-Shot Learning Recognition System")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"{'=' * 60}\n")

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model file exists and the path is correct.")
        return

    # 加载数据
    print("Loading dataset...")
    try:
        loader = AWA2Loader()
        print(f"Loaded {len(loader.test_names)} test classes")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check dataset path.")
        return

    # 初始化模型
    print("\nInitializing model...")
    n_attrs = loader.attributes.shape[1]
    model = AttributeEmbeddingModel(n_attrs)

    # 加载权重
    print(f"Loading model weights from {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Test accuracy: {checkpoint.get('test_acc', 'unknown')}%")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 初始化零样本分类器
    test_attrs = torch.FloatTensor([loader.attributes[i] for i in loader.test_idx]).to(DEVICE)
    zsl_classifier = ZeroShotClassifier(test_attrs)

    # 创建输出目录
    for d in ["recognition_results", "visualizations"]:
        os.makedirs(os.path.join(OUTPUT_PATH, d), exist_ok=True)

    # 初始化可视化器和识别系统
    visualizer = Visualizer(model, zsl_classifier, loader)
    recognition = RecognitionSystem(visualizer)

    # 菜单选择
    print("\n" + "=" * 60)
    print("Function Selection")
    print("=" * 60)
    print("1. Single Image Recognition")
    print("2. Image Recognition with Visualization")
    print("3. Continuous Image Recognition")
    print("4. Real-time Camera Recognition")
    print("5. Batch Recognition (folder)")
    print("6. Exit")

    while True:
        choice = input("\nSelect function (1-6): ").strip()

        if choice == '1':
            path = input("Enter image path: ").strip().strip('"').strip("'")
            recognition.recognize_image(path)

        elif choice == '2':
            path = input("Enter image path: ").strip().strip('"').strip("'")
            recognition.recognize_and_visualize(path)

        elif choice == '3':
            recognition.continuous_recognition()

        elif choice == '4':
            recognition.camera_recognition()

        elif choice == '5':
            folder = input("Enter folder path: ").strip().strip('"').strip("'")
            recognition.batch_recognition(folder)

        elif choice == '6':
            print("Exiting program")
            break

        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()