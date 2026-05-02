"""
动物零样本识别系统 - Streamlit Web应用
使用方法：在命令行中运行 streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# ==================== 页面配置 ====================
st.set_page_config(
    page_title="动物零样本识别系统",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==================== 配置参数 ====================
DATASET_PATH = r"C:\Users\34552\Desktop\Animals_with_Attributes2"
MODEL_PATH = r"C:\Users\34552\Desktop\DS -基于属性-属性热力激活图\models\best.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================== 动物知识库 ====================
ANIMAL_FACTS = {
    'giant_panda': ['🐼 竹子饮食 (99%)', '🐼 黑白相间的皮毛', '🐼 原产于中国', '🐼 每天进食12-16小时'],
    'zebra': ['🦓 独特的条纹图案', '🦓 时速可达65公里', '🦓 群居生活', '🦓 视力极佳'],
    'giraffe': ['🦒 最高的陆地动物', '🦒 脖子有7块椎骨', '🦒 舌头长达45厘米', '🦒 每天只睡30分钟'],
    'tiger': ['🐯 夜间猎手', '🐯 优秀的游泳者', '🐯独特的条纹', '🐯 顶级捕食者'],
    'lion': ['🦁 唯一群居的猫科动物', '🦁 狮群有15-30个成员', '🦁 每天睡18-20小时', '🦁 草原之王'],
    'elephant': ['🐘 最大的陆地哺乳动物', '🐘 象鼻有10万块肌肉', '🐘 记忆力极佳', '🐘 寿命可达70年'],
    'polar_bear': ['🐻‍❄️ 优秀的游泳者', '🐻‍❄️ 厚厚的脂肪层', '🐻‍❄️ 皮肤是黑色的', '🐻‍❄️ 能从1公里外闻到猎物'],
    'wolf': ['🐺 群体狩猎', '🐺 通过嚎叫沟通', '🐺 极佳的耐力', '🐺 食物链顶端'],
    'chimpanzee': ['🐵 使用工具', '🐵 复杂的社会结构', '🐵 与人类共享98%的DNA', '🐵 学习手语'],
    'dolphin': ['🐬 高度智能', '🐬 使用回声定位', '🐬 睁着眼睛睡觉', '🐬 复杂的沟通方式']
}
def get_animal_facts(animal_name):
    for key in ANIMAL_FACTS:
        if key in animal_name.lower():
            return ANIMAL_FACTS[key]
    return ['🌍 野生动物', '🌍 独特特征', '🌍 适应环境', '🌍 受保护物种']
# ==================== 数据加载器 ====================
@st.cache_resource
class AWA2Loader:
    def __init__(self):
        try:
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

            self.attributes = (self.attributes - self.attributes.mean(axis=0)) / (self.attributes.std(axis=0) + 1e-8)

            with open(test_classes, 'r', encoding='utf-8') as f:
                self.test_names = [line.strip() for line in f.readlines()]

            self.test_idx = [self.class_to_idx[c] for c in self.test_names]
            self.test_map = {o: i for i, o in enumerate(self.test_idx)}
            self.n_attrs = self.attributes.shape[1]
        except Exception as e:
            st.error(f"加载数据集失败: {e}")
            raise e


# ==================== 模型定义 ====================
class AttributeEmbeddingModel(nn.Module):
    def __init__(self, n_attrs, dropout_rate=0.4):
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
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.attr_head(x)
class ZeroShotClassifier(nn.Module):
    def __init__(self, test_attrs):
        super().__init__()
        self.register_buffer('test_attrs', test_attrs)
    def predict_test(self, pred_attrs):
        pred_norm = F.normalize(pred_attrs, p=2, dim=1)
        test_norm = F.normalize(self.test_attrs, p=2, dim=1)
        return torch.mm(pred_norm, test_norm.t()) / 0.1
# ==================== 识别器类 ====================
@st.cache_resource
class AnimalRecognizer:
    def __init__(self):
        self.loader = None
        self.model = None
        self.zsl_classifier = None
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.is_loaded = False
    def load_models(self):
        """加载模型"""
        try:
            self.loader = AWA2Loader()
            self.model = AttributeEmbeddingModel(self.loader.n_attrs)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model'])
            self.model = self.model.to(DEVICE)
            self.model.eval()
            test_attrs = torch.FloatTensor([self.loader.attributes[i] for i in self.loader.test_idx]).to(DEVICE)
            self.zsl_classifier = ZeroShotClassifier(test_attrs).to(DEVICE)
            self.zsl_classifier.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            st.error(f"加载模型失败: {e}")
            return False
    def predict(self, image, top_k=5):
        """预测图像"""
        if not self.is_loaded:
            return None, None
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_attr = self.model(tensor)
            logits = self.zsl_classifier.predict_test(pred_attr)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            results = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                results.append({
                    'rank': i + 1,
                    'name': self.loader.test_names[idx],
                    'probability': float(prob),
                    'confidence': f"{prob * 100:.1f}%"
                })
        return results, pred_attr.squeeze().cpu().numpy()
# ==================== 可视化函数 ====================
def create_prediction_chart(results):
    """创建预测结果图表"""
    names = [r['name'] for r in results]
    probs = [r['probability'] * 100 for r in results]

    fig = go.Figure(data=[
        go.Bar(x=probs[::-1], y=names[::-1], orientation='h',
               marker_color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'][:len(probs)],
               text=[f'{p:.1f}%' for p in probs[::-1]],
               textposition='outside')
    ])

    fig.update_layout(
        title='Top-5 Predictions',
        xaxis_title='Probability (%)',
        yaxis_title='Animal Class',
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def create_confidence_gauge(confidence):
    """创建置信度仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': "Confidence Score", 'font': {'size': 24}},
        delta={'reference': 80, 'increasing': {'color': "green"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ff6b6b'},
                {'range': [50, 75], 'color': '#ffd93d'},
                {'range': [75, 100], 'color': '#6bcf7f'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100}
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_attribute_radar(pred_attr, predicates, top_k=8):
    """创建属性雷达图"""
    if pred_attr is None or len(pred_attr) == 0:
        return None

    indices = np.argsort(np.abs(pred_attr))[-top_k:]
    values = pred_attr[indices]
    labels = [predicates[i][:15] for i in indices]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Attributes',
        line_color='#e74c3c',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[float(values.min()), float(values.max())]
            )),
        showlegend=False,
        title='Key Attribute Activation',
        height=400
    )

    return fig


# ==================== 摄像头处理类 ====================
class CameraProcessor:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.last_result = None
        self.last_time = 0

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_time > 0.5:  # 每0.5秒处理一次
            results, _ = self.recognizer.predict(frame)
            if results:
                self.last_result = results[0]
            self.last_time = current_time

        # 在图像上绘制结果
        if self.last_result:
            cv2.rectangle(frame, (5, 5), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Animal: {self.last_result['name']}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {self.last_result['confidence']}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame


# ==================== 主界面 ====================
def main():
    # 侧边栏
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1995/1995572.png", width=100)
        st.title("🐘 动物识别系统")
        st.markdown("---")

        # 初始化模型
        if 'recognizer' not in st.session_state:
            st.session_state.recognizer = AnimalRecognizer()

        if not st.session_state.recognizer.is_loaded:
            if st.button("🚀 加载模型", type="primary", use_container_width=True):
                with st.spinner("正在加载模型，请稍候..."):
                    if st.session_state.recognizer.load_models():
                        st.success("✅ 模型加载成功！")
                        st.rerun()
                    else:
                        st.error("❌ 模型加载失败")
        else:
            st.success("✅ 模型已加载")
            st.info(f"""
            📊 系统信息：
            - 设备：{DEVICE}
            - 测试类别：{len(st.session_state.recognizer.loader.test_names)} 种
            """)

        st.markdown("---")
        st.markdown("### 🎯 功能说明")
        st.markdown("""
        - **图像识别**：上传图片进行识别
        - **实时摄像头**：视频流识别
        - **批量处理**：多图片批量识别
        - **识别历史**：查看识别记录
        """)

        st.markdown("---")
        st.markdown("### 📞 关于")
        st.markdown("基于零样本学习的智能动物识别系统")

    # 主内容区
    st.title("🐘 动物零样本识别系统")
    st.markdown("基于深度学习和零样本学习的智能动物识别系统 - 支持50种动物识别")

    if not st.session_state.recognizer.is_loaded:
        st.warning("⚠️ 请先在左侧边栏点击「加载模型」按钮启动系统")
        return

    # 主标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📷 图像识别", "🎥 实时摄像头", "📊 批量处理", "📈 识别历史"])

    # Tab 1: 图像识别
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 上传图像")
            uploaded_file = st.file_uploader("选择图像文件", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的图像", use_container_width=True)

                if st.button("🔍 开始识别", type="primary", use_container_width=True):
                    with st.spinner("识别中..."):
                        results, pred_attr = st.session_state.recognizer.predict(np.array(image))

                        if results:
                            st.session_state.last_result = {
                                'image': image,
                                'results': results,
                                'pred_attr': pred_attr,
                                'timestamp': datetime.now()
                            }
                            st.success("✅ 识别完成！")
                            st.rerun()

        with col2:
            if 'last_result' in st.session_state:
                st.subheader("🎯 识别结果")

                results = st.session_state.last_result['results']
                top_result = results[0]

                # 显示主要结果
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white;">
                    <h3 style="margin: 0;">🏆 主要识别结果</h3>
                    <p style="font-size: 24px; margin: 10px 0;">{top_result['name']}</p>
                    <p style="font-size: 18px;">置信度: {top_result['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)

                # 置信度仪表盘
                gauge = create_confidence_gauge(top_result['probability'])
                st.plotly_chart(gauge, use_container_width=True)

                # 预测图表
                chart = create_prediction_chart(results)
                st.plotly_chart(chart, use_container_width=True)

                # 动物趣闻
                with st.expander("📚 动物趣闻", expanded=True):
                    facts = get_animal_facts(top_result['name'])
                    for fact in facts:
                        st.info(fact)

                # 属性雷达图
                if hasattr(st.session_state.recognizer.loader, 'predicates'):
                    radar = create_attribute_radar(
                        st.session_state.last_result['pred_attr'],
                        st.session_state.recognizer.loader.predicates
                    )
                    if radar:
                        st.plotly_chart(radar, use_container_width=True)

    # Tab 2: 实时摄像头
    with tab2:
        st.subheader("🎥 实时摄像头识别")
        st.warning("⚠️ 注意：使用摄像头需要允许浏览器访问摄像头权限")

        run_camera = st.checkbox("启动摄像头", value=False)

        if run_camera:
            processor = CameraProcessor(st.session_state.recognizer)

            # 使用OpenCV直接访问摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("无法打开摄像头，请检查摄像头连接")
            else:
                frame_placeholder = st.empty()
                stop_button = st.button("停止摄像头")

                while run_camera and not stop_button:
                    ret, frame = cap.read()
                    if ret:
                        frame = processor.process_frame(frame)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    time.sleep(0.03)

                cap.release()
                st.info("摄像头已关闭")

    # Tab 3: 批量处理
    with tab3:
        st.subheader("📊 批量图像处理")

        uploaded_files = st.file_uploader("选择多张图像", type=['jpg', 'jpeg', 'png', 'bmp'],
                                          accept_multiple_files=True)

        if uploaded_files:
            st.write(f"已选择 {len(uploaded_files)} 张图像")

            if st.button("🚀 开始批量识别", type="primary", use_container_width=True):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"正在处理: {file.name}")
                    image = Image.open(file)
                    results, _ = st.session_state.recognizer.predict(np.array(image))

                    if results:
                        batch_results.append({
                            'filename': file.name,
                            'prediction': results[0]['name'],
                            'confidence': results[0]['confidence']
                        })

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("批量识别完成！")
                st.session_state.batch_results = batch_results
                st.success(f"✅ 成功识别 {len(batch_results)}/{len(uploaded_files)} 张图像")
                st.rerun()

        # 显示批量结果
        if 'batch_results' in st.session_state and st.session_state.batch_results:
            st.subheader("📋 批量识别结果")

            # 创建结果表格
            result_df = pd.DataFrame(st.session_state.batch_results)
            st.dataframe(result_df, use_container_width=True)

            # 导出结果
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="💾 导出结果 (CSV)",
                data=csv,
                file_name=f"recognition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Tab 4: 识别历史
    with tab4:
        st.subheader("📈 识别历史记录")

        if 'recognition_history' not in st.session_state:
            st.session_state.recognition_history = []

        # 添加当前结果到历史
        if 'last_result' in st.session_state:
            current_time = datetime.now()
            if not st.session_state.recognition_history or \
                    (current_time - st.session_state.recognition_history[-1]['timestamp']).seconds > 5:
                st.session_state.recognition_history.append({
                    'timestamp': current_time,
                    'animal': st.session_state.last_result['results'][0]['name'],
                    'confidence': st.session_state.last_result['results'][0]['confidence']
                })

        # 显示统计信息
        if st.session_state.recognition_history:
            col1, col2, col3 = st.columns(3)
            history_df = pd.DataFrame(st.session_state.recognition_history)

            with col1:
                st.metric("总识别次数", len(history_df))

            with col2:
                history_df['confidence_num'] = history_df['confidence'].str.rstrip('%').astype(float)
                avg_conf = history_df['confidence_num'].mean()
                st.metric("平均置信度", f"{avg_conf:.1f}%")

            with col3:
                unique_animals = history_df['animal'].nunique()
                st.metric("识别动物种类", unique_animals)

            # 识别历史图表
            if len(history_df) > 1:
                fig = px.line(history_df, x='timestamp', y='confidence_num',
                              title='识别置信度趋势',
                              labels={'timestamp': '时间', 'confidence_num': '置信度 (%)'})
                st.plotly_chart(fig, use_container_width=True)

            # 识别历史表格
            st.dataframe(history_df[['timestamp', 'animal', 'confidence']].tail(20),
                         use_container_width=True)

            # 清空历史
            if st.button("🗑️ 清空历史", use_container_width=True):
                st.session_state.recognition_history = []
                st.rerun()
        else:
            st.info("暂无识别历史，请先进行图像识别")

    # 统计图表
    if 'recognition_history' in st.session_state and st.session_state.recognition_history:
        with st.expander("📊 详细统计分析"):
            history_df = pd.DataFrame(st.session_state.recognition_history)
            history_df['confidence_num'] = history_df['confidence'].str.rstrip('%').astype(float)

            # 识别分布饼图
            animal_counts = history_df['animal'].value_counts()
            fig_pie = px.pie(values=animal_counts.values, names=animal_counts.index,
                             title='识别动物分布')
            st.plotly_chart(fig_pie, use_container_width=True)


if __name__ == "__main__":
    main()