import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image
import torch
from io import BytesIO

# 设置页面配置
st.set_page_config(page_title="Image to Video Converter", page_icon='')

# 应用头部
st.header("Convert your Photos into Videos")

# 文件上传器
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # 显示上传的图片
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # 加载模型
    model_name = "NimVideo/cogvideox-2b-img2vid"
    pipe = DiffusionPipeline.from_pretrained(model_name)
    pipe.to('cuda')
    
    # 将上传的文件转换为PIL图像
    image = Image.open(BytesIO(uploaded_file.read()))
    
    # 调用模型生成视频
    try:
        # 根据模型要求，设置prompt
        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        generated_image = pipe(prompt).images[0]
        
        # 显示生成的图片
        st.image(generated_image, caption='Generated Video Frame.', use_column_width=True)
    except Exception as e:
        st.error(f"Error generating video: {str(e)}")
