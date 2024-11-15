# import streamlit as st # type: ignore
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import gradio as gr
from scipy.io import wavfile

# number=0

def musicgen(img,text):
    # 加载 BLIP 图像描述生成模型
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

    # 加载音乐生成模型
    music_processor = AutoProcessor.from_pretrained(r"facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained(r"facebook/musicgen-small")
    # 加载图像
    raw_image = img.convert('RGB')

    # 有条件图像描述生成
    # text = "a photograph of"
    inputs = blip_processor(raw_image, text, return_tensors="pt").to("cpu")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # 无条件图像描述生成
    # inputs = blip_processor(raw_image, return_tensors="pt").to("cuda")
    # out = blip_model.generate(**inputs)
    # caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # 根据描述生成音乐
    music_inputs = music_processor(
        text=caption,
        padding=True,
        return_tensors="pt",
    )
    audio_values = music_model.generate(**music_inputs, max_new_tokens=256)

    # 将音频数据类型转换为 numpy.int16
    audio_data = audio_values[0,0].numpy()
    # 保存音频文件
    # number+=1
    audio_filename = r".\music\generated_audio.wav"
    wavfile.write(audio_filename, music_model.config.audio_encoder.sampling_rate, audio_data)
    audio = wavfile.read(r".\music\generated_audio.wav")
    return audio,caption

with gr.Blocks(gr.themes.Soft()) as demo:
    # gr.Markdown('音乐生成')
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image=gr.Image(sources=['upload'],label="上传图片",type='pil')
                gr.Examples([r'.\image_test\test1.jpg',r'.\image_test\test2.jpg',r'.\image_test\test3.png',r'.\image_test\test4.jpg'],label="参考范例",inputs=[input_image])
                button=gr.Button(value='生成',variant="primary")
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2,label="文本提示")
                output_music=gr.Audio(type="numpy")
    
    button.click(musicgen,[input_image,captioning],[output_music,captioning])



demo.launch()