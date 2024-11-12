# import streamlit as st # type: ignore
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import gradio as gr
from scipy.io import wavfile
import torchaudio
from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

def musicgen(img,text):
    # 加载 BLIP 图像描述生成模型
    blip_processor = BlipProcessor.from_pretrained(r"D:\Work\blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(r"D:\Work\blip-image-captioning-large").to("cuda")

    # 加载音乐生成模型
    # music_processor = AutoProcessor.from_pretrained(r"D:\Work\musicgen-melody")
    # music_model = MusicgenForConditionalGeneration.from_pretrained(r"D:\Work\musicgen-melody")
    music_model = MAGNeT.get_pretrained("facebook/magnet-small-30secs")
    # music_model.set_generation_params(duration=dura)

    # 加载图像
    raw_image = img.convert('RGB')

    # 有条件图像描述生成
    # text = "a photograph of"
    inputs = blip_processor(raw_image, text, return_tensors="pt").to("cuda")
    print("blip_processor finished.")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # 无条件图像描述生成
    # inputs = blip_processor(raw_image, return_tensors="pt").to("cuda")
    # out = blip_model.generate(**inputs)
    # caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # 根据描述生成音乐
    # music_inputs = music_processor(
    #     text=caption,
    #     duration=dura,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # audio_values = music_model.generate(**music_inputs, max_new_tokens=256)
    audio_values = music_model.generate(caption)

    for idx,one_wav in enumerate(audio_values):
        audio_write(f'{idx}',one_wav.cpu(),music_model.sample_rate,strategy="loudness")

    # # 将音频数据类型转换为 numpy.int16
    # audio_data = audio_values[0,0].numpy()
    # # 保存音频文件
    # audio_filename = "generated_audio.wav"
    # wavfile.write(audio_filename, music_model.config.audio_encoder.sampling_rate, audio_data)
    audio = wavfile.read("generated_audio.wav")
    return audio,caption

with gr.Blocks(gr.themes.Soft()) as demo:
    gr.Markdown('音乐生成')
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image=gr.Image(sources=['upload'],label="上传图片",type='pil')
                # dura = gr.Slider(minimum=5,maximum=60,label="音乐时长",step=5)
                gr.Examples([r'D:\Work\image_test\test1.jpg',r'D:\Work\image_test\test2.jpg',r'D:\Work\image_test\test3.png',r'D:\Work\image_test\test4.jpg'],inputs=[input_image])
                button=gr.Button(value='生成',variant="primary")
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2,label="文本提示")
                output_music=gr.Audio(type="numpy")
    
    button.click(musicgen,[input_image,captioning],[output_music,captioning])



demo.launch()