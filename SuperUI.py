import streamlit as st
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# streamlit run SuperUI.py --server.address 0.0.0.0
# 设置页面
st.set_page_config(page_title="SonicMaster 音频复原对比工具", layout="wide")

# --- 样式优化 ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stAudio {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner="正在生成频谱图...")
def generate_spectrogram(audio_path, title):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # 计算梅尔频谱
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set(title=f'Spectrogram: {title}')
        return fig
    except Exception as e:
        return None

# --- 侧边栏：路径配置 ---
st.sidebar.header("📁 路径配置")
# 默认路径可以根据你的常规输出目录修改
base_path = st.sidebar.text_input("根目录 (Output Dir)", "/inspire/hdd/global_user/chenxie-25019/HaoQiu/RESULT/Audio_output")
restored_dir = st.sidebar.text_input("1. 复原后 (Restored)", os.path.join(base_path, "restored_audio"))
degraded_dir = st.sidebar.text_input("2. 退化前 (Degraded)", os.path.join(base_path, "degraded_audio"))
original_dir = st.sidebar.text_input("3. 原始 (Ground Truth)", os.path.join(base_path, "original_audio"))

# --- 检查路径并获取文件名列表 ---
valid_files = []
if os.path.exists(original_dir):
    valid_files = sorted([f for f in os.listdir(original_dir) if f.endswith(('.flac', '.wav', '.mp3'))])

st.title("🎼 SonicMaster 音频复原对比看板")

if not valid_files:
    st.info("请在侧边栏配置正确的文件夹路径。脚本将以 '原始 (Target)' 文件夹作为基准进行匹配。")
else:
    # --- 选择音频 ---
    st.header("🔍 选择测试样本")
    col_sel, col_ran = st.columns([3, 1])
    
    with col_sel:
        selected_file = st.selectbox("选择一个音频文件进行对比:", valid_files)
    with col_ran:
        st.write(" ")
        if st.button("🔀 随机抽取"):
            selected_file = random.choice(valid_files)
            st.rerun()

    st.divider()

    # --- 三列对比模型 ---
    # 定义展示内容：标题、文件夹路径、配色方案
    display_info = [
        {"title": "❌ 退化音频 (Input/Degraded)", "path": degraded_dir, "color": "Reds"},
        {"title": "✨ 模型复原 (Result/Inference)", "path": restored_dir, "color": "Blues"},
        {"title": "✅ 原始完美音频 (Target/GT)", "path": original_dir, "color": "Greens"}
    ]

    cols = st.columns(3)
    
    for i, info in enumerate(display_info):
        with cols[i]:
            st.subheader(info["title"])
            file_path = os.path.join(info["path"], selected_file)
            
            if os.path.exists(file_path):
                st.caption(f"文件名: {selected_file}")
                # 播放器
                st.audio(file_path)
                # 频谱图
                fig = generate_spectrogram(file_path, selected_file)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("频谱图生成失败")
            else:
                st.warning(f"文件不存在: \n`{file_path}`")

    st.success(f"当前对比样本 ID: {Path(selected_file).stem}")