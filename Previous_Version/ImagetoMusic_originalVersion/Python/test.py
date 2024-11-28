import streamlit as st # type: ignore
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile # type: ignore
import numpy as np
import gradio as gr


def musicgen(img,text):
    
    audio = wavfile.read(r".\music\generate.wav")

    return audio,text

with gr.Blocks(gr.themes.Soft()) as demo:
    gr.Markdown('音乐生成')
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image=gr.Image(sources=['upload'],label="上传图片",type='pil')
                gr.Examples([r'D:\Work\image_test\test1.jpg',r'D:\Work\image_test\test2.jpg',r'D:\Work\image_test\test3.png',r'D:\Work\image_test\test2.jpg'],inputs=[input_image])
                button=gr.Button(value='生成',variant="primary")
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2,label="文本提示")
                output_music=gr.Audio(type="numpy")
    
    button.click(musicgen,[input_image,captioning],[output_music,captioning])



demo.launch()