import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline, AudioLDM2Pipeline
from transformers import pipeline as hf_pipeline
import torch
import soundfile as sf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

# 1. 加载模型
@st.cache_resource
def load_models():
    # 加载Stable Diffusion模型，用于图片风格融合
    sd_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
    # 加载图片描述生成模型（CLIP或其他图片描述模型）
    caption_generator = hf_pipeline("image-captioning")
    # 加载AudioLDM模型，用于音频生成
    audio_pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16).to("cuda")
    return sd_pipeline, caption_generator, audio_pipeline

# 初始化加载的模型
sd_pipeline, caption_generator, audio_pipeline = load_models()

# 2. 用户上传照片模块
st.title("城市涂鸦纪念品生成器")
st.write("上传您的3张涂鸦照片，我们将为您生成专属纪念品。")

# 上传图片，限制为3张
uploaded_files = st.file_uploader("请上传3张涂鸦照片", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
if uploaded_files and len(uploaded_files) == 3:
    st.write("您上传的照片预览：")
    images = [Image.open(file) for file in uploaded_files]
    for img in images:
        st.image(img, caption="上传的照片")

    # 3. 使用Stable Diffusion进行图片融合
    def merge_and_style_images(images, prompt="graffiti style, urban street art"):
        # 先将3张图片横向拼接在一起
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)
        combined_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width
        # 使用Stable Diffusion对拼接的图像进行风格化
        styled_image = sd_pipeline(prompt=prompt, init_image=combined_image, strength=0.75, guidance_scale=7.5).images[0]
        styled_image.save("styled_postcard.jpg")
        return styled_image

    # 生成风格化的纪念图片
    styled_image = merge_and_style_images(images)
    st.image(styled_image, caption="生成的涂鸦明信片")

    # 4. 从图片生成文字描述
    description = caption_generator(styled_image)[0]['caption']
    st.write("生成的图片描述:", description)

    # 5. 基于描述生成音频
    def generate_audio_from_text(description):
        audio_array = audio_pipeline(prompt=description).audios[0]
        sf.write("generated_audio.wav", audio_array, samplerate=16000)  # 将音频保存为wav格式
        return "generated_audio.wav"

    # 生成的音频文件路径
    audio_file_path = generate_audio_from_text(description)
    st.audio(audio_file_path, format="audio/wav")

    # 6. 下载和可选的邮件发送模块
    # 提供用户下载生成的图片和音频
    st.download_button("下载生成的明信片", data=open("styled_postcard.jpg", "rb").read(), file_name="postcard.jpg")
    st.download_button("下载生成的音频", data=open(audio_file_path, "rb").read(), file_name="audio.wav")

    # 可选的邮件发送函数
    def send_email_with_attachments(email_address, image_path, audio_path):
        msg = MIMEMultipart()
        msg['From'] = 'your_email@example.com'
        msg['To'] = email_address
        msg['Subject'] = "您的涂鸦之旅纪念品"
        
        # 附加图片文件
        with open(image_path, "rb") as img_file:
            img = MIMEBase('application', 'octet-stream')
            img.set_payload(img_file.read())
            encoders.encode_base64(img)
            img.add_header('Content-Disposition', f'attachment; filename="postcard.jpg"')
            msg.attach(img)
        
        # 附加音频文件
        with open(audio_path, "rb") as audio_file:
            audio = MIMEBase('application', 'octet-stream')
            audio.set_payload(audio_file.read())
            encoders.encode_base64(audio)
            audio.add_header('Content-Disposition', f'attachment; filename="audio.wav"')
            msg.attach(audio)

        # 发送邮件
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.login('your_email@example.com', 'password')
            server.sendmail(msg['From'], msg['To'], msg.as_string())
    
    # 用户输入邮箱地址并发送
    email = st.text_input("输入您的邮箱以接收纪念品")
    if email and st.button("发送至邮箱"):
        send_email_with_attachments(email, "styled_postcard.jpg", "generated_audio.wav")
        st.success("邮件已成功发送！")

else:
    st.warning("请确保上传3张图片。")

