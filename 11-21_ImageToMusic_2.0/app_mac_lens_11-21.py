import streamlit as st  # type: ignore
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import gradio as gr
from scipy.io import wavfile

def musicgen(img, text):
    global number
    number = 0

    # 加载 BLIP 图像描述生成模型
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("mps")

    # 加载音乐生成模型
    music_processor = AutoProcessor.from_pretrained(r"facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained(r"facebook/musicgen-small")

    # 加载图像
    raw_image = img.convert('RGB')

    # 有条件图像描述生成
    text = "a photograph of"
    inputs = blip_processor(raw_image, text, return_tensors="pt").to("mps")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # 无条件图像描述生成
    inputs = blip_processor(raw_image, return_tensors="pt").to("mps")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # 根据描述生成音乐
    music_inputs = music_processor(
        text=caption,
        padding=True,
        return_tensors="pt",
    )
    audio_values = music_model.generate(**music_inputs, max_new_tokens=256)

    # 将音频数据类型转换为 numpy.int16
    audio_data = audio_values[0, 0].numpy()

    # 保存音频文件
    number += 1
    audio_filename = r".\music\generated_audio.wav"
    wavfile.write(audio_filename, music_model.config.audio_encoder.sampling_rate, audio_data)
    audio = wavfile.read(r".\music\generated_audio.wav")
    return audio, caption

# Memphis Style CSS with geometric shapes and background image
css = """
body {
    background-image: url('https://cdn.discordapp.com/attachments/1212332093456257036/1309096379737440286/backdrop.png?ex=67405682&is=673f0502&hm=e4e48ee7287247d1b64758955f445094c6a9d56e3f1a8a937a0c6cc5703366ef&');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;  /* Optional: change text color for contrast */
}
.gradio-container {
    position: relative;
    z-index: 10;
    padding: 20px;
    background: rgba(0, 0, 0, 0.5); /* Optional: add semi-transparent background to enhance readability */
    border-radius: 10px;
}
"""

# Create Gradio interface
with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(sources=['upload', 'webcam'], label="Upload pictures or use the camera to shoot", type='pil')
                # Use URL links for the example images
                gr.Examples(
                    [
                        'https://cdn.discordapp.com/attachments/1212332093456257036/1309103257599082526/image.png?ex=67405cea&is=673f0b6a&hm=28a21f3bd60fa41aac95da8109c019f9e22ee7a5d5c724e005b8c39c6aa0f206&',  # 图片1的URL
                        'https://cdn.discordapp.com/attachments/1212332093456257036/1309101622802255965/30e6ec9911132e5d.jpeg?ex=67405b64&is=673f09e4&hm=32c585bc50c641baf657305818c383200a17e99a4f71492bb4ab35cfd588d828&',  # 图片2的URL
                        'https://cdn.discordapp.com/attachments/1212332093456257036/1309101955179614288/image.png?ex=67405bb4&is=673f0a34&hm=99ddcaeed5cdbee72c71b7f23459d0fabddc10b577b305d350d0f38a23384d8d&',  # 图片3的URL
                        'https://cdn.discordapp.com/attachments/1212332093456257036/1309102165263908864/image.png?ex=67405be6&is=673f0a66&hm=f241f21367fc77ab2fd47812ec1d1e6435f8f266282cf87b798123c711fc8d31&'   # 图片4的URL
                    ],
                    label="Sample Images", inputs=[input_image]
                )
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2, label="Prompt")
                output_music = gr.Audio(type="numpy")
        button = gr.Button(value='Generate', variant="primary")
    
    # Button click event
    button.click(musicgen, [input_image, captioning], [output_music, captioning])

    # Apply custom CSS
    demo.css = css


demo.launch()
