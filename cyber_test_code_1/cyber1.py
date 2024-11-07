import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
import numpy as np

# 设置GPU设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载Stable Diffusion XL模型和IP-Adapter
@st.cache_resource
def load_model():
    # 加载Stable Diffusion XL模型并插入IP-Adapter
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-1024-v1-0",
        revision="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    # 加载文本编码器（CLIP）以处理输入的文本提示
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch32").to(device)
    
    return pipe, tokenizer, text_encoder

# 定义视频处理类
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_image = None  # 用于存储识别到的人脸图像
        self.pipe, self.tokenizer, self.text_encoder = load_model()
        self.prompt = ""

    def transform(self, frame):
        # 获取图像数据并进行灰度转换
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # 截取第一个检测到的人脸
            x, y, w, h = faces[0]
            self.face_image = img[y:y+h, x:x+w]

        return img

    def generate_image(self):
        # 如果有检测到的人脸，则进行风格化处理
        if self.face_image is not None:
            # 转为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(self.face_image, cv2.COLOR_BGR2RGB))

            # 对输入文本进行处理
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(device)
            text_embeddings = self.text_encoder(**inputs).last_hidden_state

            # 使用Stable Diffusion XL进行风格转换
            with torch.no_grad():
                result = self.pipe(prompt_embeds=text_embeddings, init_image=pil_image).images[0]

            # 将结果保存为PNG
            result.save("cyberpunk_face.png")

            return result
        else:
            return None

# Streamlit UI部分
st.title("赛博风格化人脸生成器")

# 显示实时视频流并进行人脸识别
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# 获取用户输入的文本提示
prompt = st.text_input("请输入赛博风格化的文本提示：", "cyberpunk portrait")

# 生成赛博风格化图像并显示
video_transformer = VideoTransformer()
video_transformer.prompt = prompt

if st.button("生成赛博风格化图像"):
    result_image = video_transformer.generate_image()
    
    if result_image:
        st.image(result_image, caption="赛博风格化的图像", use_column_width=True)
        st.download_button(
            label="下载赛博风格化图像",
            data=open("cyberpunk_face.png", "rb").read(),
            file_name="cyberpunk_face.png",
            mime="image/png"
        )
    else:
        st.warning("未检测到人脸，请确保视频中有清晰的人脸。")