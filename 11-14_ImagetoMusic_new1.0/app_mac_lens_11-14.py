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


# Memphis Style CSS with geometric shapes
css = """
    .gradio-container {
        background-color: #f8f8f8;
        color: #333;
        font-family: 'Comic Sans MS', sans-serif;
        position: relative;
    }
    
    .gr-button {
        background-color: #FF6347;  /* Tomato */
        border-radius: 10px;
        color: white;
        font-weight: bold;
        padding: 15px;
        text-transform: uppercase;
    }
    
    .gr-button:hover {
        background-color: #FF4500;  /* OrangeRed */
    }

    .gr-column {
        background-color: #FFFACD;  /* LemonChiffon */
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .gr-textbox {
        background-color: #F0E68C;  /* Khaki */
        border-radius: 8px;
        border: 2px solid #FF6347;
    }

    .gr-markdown {
        font-size: 24px;
        font-weight: bold;
        color: #32CD32;  /* LimeGreen */
        text-align: center;
    }

    .gr-image {
        border: 4px dashed #FF6347;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }

    .gr-audio {
        background-color: #ADD8E6;  /* LightBlue */
        padding: 10px;
        border-radius: 10px;
    }

    .gr-examples {
        margin-top: 20px;
        font-weight: bold;
        color: #FF4500;
    }

    .gr-row {
        background-color: #FFE4B5;  /* Moccasin */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Geometric shapes */
    .gradio-container::before {
        content: '';
        position: absolute;
        top: 10%;
        left: 20%;
        width: 200px;
        height: 200px;
        background-color: rgba(255, 99, 71, 0.3);
        border-radius: 50%;
        box-shadow: 0 0 30px rgba(255, 99, 71, 0.5);
    }

    .gradio-container::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 60%;
        width: 150px;
        height: 150px;
        background-color: rgba(135, 206, 235, 0.3);
        clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
        box-shadow: 0 0 30px rgba(135, 206, 235, 0.5);
    }
"""

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown('### Music Generation')
    
    with gr.Row():
        with gr.Column(scale=4):
            # Enable both file upload and camera
        
            input_image = gr.Image(sources=['upload', 'webcam'], label="上传图片或使用摄像头拍摄", type='pil')

            gr.Examples([r'.\image_test\test1.jpg', r'.\image_test\test2.jpg', r'.\image_test\test3.png', r'.\image_test\test4.jpg'], 
                        label="Sample Images", inputs=[input_image])
        with gr.Column(scale=4):
            captioning = gr.Textbox(lines=2, label="Text")
            output_music = gr.Audio(type="numpy")
    
    button = gr.Button(value='Generate', variant="primary")
    
    button.click(musicgen, [input_image, captioning], [output_music, captioning])

demo.launch()
