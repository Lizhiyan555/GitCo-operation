import streamlit as st
import requests
import uuid

# 设置页面配置
st.set_page_config(page_title="Image to Music Creator", page_icon='')

# 应用头部
st.header("Turn your Photos into Music")
st.markdown("1. Select a photo from your pc\n 2. AI detects the photo description\n3. AI generates music based on the photo")

# 文件上传器
image_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])
if image_file is not None:
    # 显示上传的图片
    st.image(image_file, caption="Uploaded Image...", use_column_width=True)

    # 调用AI模型生成音乐
    audio_file = generate_music_from_image(image_file)
    if audio_file:
        st.audio(audio_file, caption="Generated Music")

def generate_music_from_image(image_file):
    # 这里需要实现将图片转换为音乐的逻辑
    # 可以使用Hugging Face上的Image to Music模型，或者任何其他API服务
    # 以下代码仅为示例，需要根据实际API进行调整
    API_URL = "https://huggingface.co/spaces/fffiloni/image-to-music-v2"
    headers = {"Authorization": "Bearer YOUR_API_TOKEN"}
    files = {'image': image_file}
    response = requests.post(API_URL, headers=headers, files=files)
    if response.status_code == 200:
        return response.content  # 返回音频文件内容
    else:
        return None
